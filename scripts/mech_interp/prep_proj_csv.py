import pandas as pd
import json
import fire
import os
from loguru import logger

def main(fp: str, output_dir: str, persona_type: str):
    assert fp.endswith(".jsonl")
    prompts = []
    answers = []

    if "qwen2.5" in fp:
        with open(fp, "r") as rf:
            for line in rf:
                obj = json.loads(line)

                prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + obj["prompt"] + "<|im_start|>assistant"
                response = obj["final_answer"]
                prompts.append(prompt)
                answers.append(response)
    
    elif "s1.1" in fp:
        with open(fp, "r") as rf:
            for line in rf:
                obj = json.loads(line)

                prompt = obj["prompt"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant"
                response = obj["prompt"].split("<|im_start|>assistant")[1] + obj['cot'] + "\n<|im_start|>answer\n" + obj["final_answer"]
                prompts.append(prompt)
                answers.append(response)

    df = pd.DataFrame({"prompt": prompts, "answer": answers})
    saved_fp = os.path.join(output_dir, os.path.basename(fp).replace(".jsonl", f"_{persona_type}.csv"))
    df.to_csv(saved_fp, index=False)
    logger.success(f"Saved prompt-answer pairs from jsonl to csv: {saved_fp}")

if __name__ == "__main__":
    fire.Fire(main)