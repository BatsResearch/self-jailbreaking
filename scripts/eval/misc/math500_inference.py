from datasets import load_dataset
######
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import pathlib
import torch
import argparse
import os
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='S1 Model Inference')
parser.add_argument('--output-filename', type=str, default=None, help='Custom output filename (default: s1_outputs_<timestamp>.jsonl)')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
parser.add_argument('--max-tokens-thinking', type=int, default=8000, help='Max tokens for thinking')
parser.add_argument('--max-tokens-answer', type=int, default=500, help='Max tokens for answer')
parser.add_argument('--num-ignore', type=int, default=0, help='Number of times to ignore end-of-thinking token')
args = parser.parse_args()

# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = args.max_tokens_thinking
MAX_TOKENS_ANSWER = args.max_tokens_answer
NUM_IGNORE = args.num_ignore
MODEL_PATH = args.model_path
MODEL_NAME = MODEL_PATH.split("/")[-2] + "__" + MODEL_PATH.split("/")[-1]

NGPU = torch.cuda.device_count()

model = LLM(
    MODEL_PATH, # s1 originally gets this prompt wrong but with budget forcing it fixes it
    tensor_parallel_size=NGPU,
    dtype=torch.bfloat16
)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
stop_token_ids = tok("<|im_end|>")["input_ids"]

datasets = "HuggingFaceH4/MATH-500"
dataset = load_dataset(datasets, split="test")
prompts = dataset['problem']
answers = dataset['answer']

SYS_PROMPT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Prepare output directory and file if saving is enabled
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

if args.output_filename:
    output_file = os.path.join(output_dir, args.output_filename)
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{MODEL_NAME}_outputs_{timestamp}.jsonl')

wf = open(output_file, 'w', encoding='utf-8', buffering=1)


########################################################
# adapted from the README of https://github.com/simplescaling/s1
# TODO: can be optimized for batch inference
########################################################

for prompt_i, p in tqdm(enumerate(prompts)):
    prompt = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_THINKING,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    prompt += "<|im_start|>think\n"
    o = model.generate(
        prompt,
        sampling_params=sampling_params
    )
    ignore_str = "Wait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    if max_tokens_thinking_tmp > 0:
        for i in range(NUM_IGNORE): # Num of times to skip stop token
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp > 0:
                prompt += o[0].outputs[0].text + ignore_str
                sampling_params = SamplingParams(
                    max_tokens=max_tokens_thinking_tmp,
                    min_tokens=1,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=0.0,
                )
                o = model.generate(
                    prompt,
                    sampling_params=sampling_params
                )
    
    prompt += o[0].outputs[0].text
    
    ### Final answer ###
    prompt += "\n<|im_start|>answer\nFinal Answer: \\boxed" # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting

    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_ANSWER,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )
    final_output = prompt + o[0].outputs[0].text

    # Save output if enabled
    output_entry = {
        "prompt_index": prompt_i,
        "user_prompt": prompts[prompt_i],
        "answer": answers[prompt_i],
        "full_conversation": final_output,
        "final_answer": o[0].outputs[0].text
    }
    wf.write(json.dumps(output_entry, ensure_ascii=False) + '\n')


# Save all outputs to file if enabled
logger.success(f"\nOutputs saved to: {output_file}")


