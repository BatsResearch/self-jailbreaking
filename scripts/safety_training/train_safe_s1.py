from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import torch
from datetime import datetime
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=float, default=0.01, description="Sample size of the safety dataset")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", description="Base model")
parser.add_argument("--output_dir", type=str, default="./outputs/safe-s1")
args = parser.parse_args()


SAMPLE_SIZE = args.sample_size # 50% of the safety dataset
BASE_MODEL = args.base_model
OUTPUT_DIR = args.output_dir

# Load dataset - using tokenized version as in s1_train.sh
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
safety_dataset = load_dataset("UCSC-VLAA/STAR-1", split="train")


# process safety dataset so it has the same format as the dataset
def to_s1k_format_safety(example):
    prompt = example['question']
    cot_answer = example['response']
    cot, answer = cot_answer.split("</think>")

    answer = answer.strip()
    cot = cot.removeprefix("<think>").strip()

    return {
        "question": prompt,
        "deepseek_attempt": answer,
        "deepseek_thinking_trajectory": cot
    }

safety_dataset = safety_dataset.map(to_s1k_format_safety)
print(safety_dataset[0])

# merge datasets
dataset = concatenate_datasets([dataset, safety_dataset.select(range(int(len(safety_dataset) * SAMPLE_SIZE)))])

########################################################
base_model = BASE_MODEL
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_s1_format(example):
    # Change 'input' and 'output' to the actual field names as necessary
    user_content = example['question']
    assistant_content = example['deepseek_attempt']
    thinking = example.get('deepseek_thinking_trajectory', None)   # Change if there's a different field for 'thinking'

    FORMAT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n{thinking}<|im_end|>\n<|im_start|>answer\n{answer}<|im_end|>"
    
    return FORMAT.format(question=user_content, thinking=thinking, answer=assistant_content)
    
response_template = "<|im_end|>\n<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

########################################################
# Training hyperparameters matching s1_train.sh
uid = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{OUTPUT_DIR}/s1-safe-7b-sample{SAMPLE_SIZE}-{uid}"
lr = 1e-5
min_lr = 0
epochs = 5
weight_decay = 1e-4
micro_batch_size = 1
gradient_accumulation_steps = 1
block_size = 32768

# Model configuration for Qwen-7B
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto",
)

# Load model
model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

# Training configuration matching s1_train.sh parameters
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    learning_rate=lr,
    weight_decay=weight_decay,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    bf16=True,
    logging_steps=1,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=3,
    max_seq_length=block_size,
    push_to_hub=False,
    save_only_model=True,
    report_to=None,

    ### memory optimization (cuda OOM error)
    gradient_checkpointing=True,
    dataloader_pin_memory=False,  # Can help with memory
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=to_s1_format,
    data_collator=collator,
)

# Start training
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)