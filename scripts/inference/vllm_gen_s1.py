"""
This script is adapted from https://github.com/simplescaling/s1
"""

from vllm import LLM, SamplingParams
import argparse
import torch
import json
from loguru import logger
from transformers import AutoTokenizer, set_seed
import pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="simplescaling/s1.1-7B", help="Name or path of the model to use")
parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
parser.add_argument("--max_think_tokens", type=int, default=500, help="Decide on a token limit for thinking generations")
parser.add_argument("--max_answer_tokens", type=int, default=200, help="Decide on a token limit for answer generations")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output files")
parser.add_argument("--data_file", type=str, default="./data/strongreject.jsonl", help="Path to safety input data file")
args = parser.parse_args()
set_seed(args.seed)

MODEL_NAME = args.model_name
MAX_THINK_TOKENS = args.max_think_tokens
MAX_ANS_TOKENS = args.max_answer_tokens
SAFETY_FP = args.data_file
OUTPUT_DIR = args.output_dir
OUTPUT_FILE_JSONL = f"{MODEL_NAME.split('/')[-1]}-think{args.max_think_tokens}-answer{args.max_answer_tokens}.jsonl"

MODEL_CONFIGS = json.load(open("./scripts/inference/model_configs.json")).get(args.model_name, {})
assert len(MODEL_CONFIGS) > 0, f"Model configuration not found for {args.model_name}"
if "system_prompt" in MODEL_CONFIGS:
    SYSTEM_PROMPT = MODEL_CONFIGS["system_prompt"]
else:
    SYSTEM_PROMPT = ""

TEMPERATURE = MODEL_CONFIGS.get("temperature", 0.0)
TOP_P = MODEL_CONFIGS.get("top_p", 1.0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = LLM(
	model=MODEL_NAME, 
	tensor_parallel_size=torch.cuda.device_count(),
	dtype=torch.bfloat16,
	seed=args.seed,
	trust_remote_code=True)

### load prompts
prompts = list()
with open(SAFETY_FP) as rf:
	for line in rf:
		line = json.loads(line)
		prompts.append(line["input_prompt"])

think_sampling_params = SamplingParams(
	max_tokens=args.max_think_tokens,
	temperature=TEMPERATURE,
	top_p=TOP_P,
	stop_token_ids=tokenizer("<|im_start|><|im_end|>")["input_ids"],
)

answer_sampling_params = SamplingParams(
	max_tokens=MAX_ANS_TOKENS,
	temperature=TEMPERATURE,
	top_p=TOP_P,
	stop_token_ids=tokenizer("<|im_end|>")["input_ids"],
)

input_prompts = list()
cots = list()
answers = list()
for p in prompts:
	prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
	prompt += "<|im_start|>think\n"
	input_prompts.append(prompt)

### Generate CoT ###
o = llm.generate(
	input_prompts,
	sampling_params=think_sampling_params
)
cots = []
new_inputs = []
for i, out in enumerate(o):
	cot = out.outputs[0].text
	if "</think>" in cot:
		cot = cot.split("</think>")[0].strip() + "</think>"
	else:
		cot = cot + "</think>" # force model to stop thinking
	
	cots.append(cot)
	new_inputs.append(input_prompts[i] + cot + "\n<|im_start|>answer\n")

	
### Final answer ###
answers = llm.generate(
	new_inputs,
	sampling_params=answer_sampling_params,
)
answers = [o.outputs[0].text for o in answers]
	
# Prepare output directory and file
SAVE_FP = pathlib.Path(OUTPUT_DIR)
SAVE_FP.mkdir(parents=True, exist_ok=True)

write_fp = SAVE_FP / OUTPUT_FILE_JSONL
logger.info(f"Writing to {write_fp}")

with open(write_fp, 'w') as f:
	for i in range(len(prompts)):
		f.write(json.dumps({
			"prompt_sent_id": prompts[i],
			"raw_prompt": prompts[i],
			"cot": cots[i],
			"final_answer": answers[i],
		}, ensure_ascii=False) + '\n')