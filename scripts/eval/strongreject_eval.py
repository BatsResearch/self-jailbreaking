import json
import multiprocessing
import os
import pathlib
import re
import warnings
from typing import Callable

import re
import openai
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pathlib
import pandas as pd
from tqdm.asyncio import tqdm

import asyncio
import termplotlib as tpl


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="Path to input file containing model outputs")
parser.add_argument("--override", action="store_true", help="override existing outputs")
args = parser.parse_args()
INPUT_FP = pathlib.Path(args.input_file)

from openai import OpenAI
client = openai.AsyncOpenAI()
OAI_MODEL = "gpt-5"



OUTPUT_DIR = INPUT_FP.parent / "oai_aisi_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FP = OUTPUT_DIR / f"oai_aisi-gpt5-labeled_{INPUT_FP.stem}.jsonl"


if OUTPUT_FP.exists() and not args.override:
    with open(OUTPUT_FP, "r", encoding="utf-8") as f:
        existing_lines = sum(1 for _ in f)
    print(f"{OUTPUT_FP} exists, found {existing_lines} lines, will skip them.")
    wf = open(OUTPUT_FP, "a", encoding="utf-8", buffering=1)
else:
    existing_lines = 0
    wf = open(OUTPUT_FP, "w", encoding="utf-8", buffering=1)


print(f"Saving to {OUTPUT_FP}, will skip {existing_lines} lines.")



########################################################
JUDGE_TEMPLATES = json.load(open("./data/sr_judge_templates.json"))

async def oai_eval(input_text):
    try:
        response = await client.responses.create(
        model=OAI_MODEL,
            input=input_text
        )
    except Exception as e:
        print(f"Error evaluating input: {e}")
        return "<score>-1</score>"
    return response.output_text

def extract_score(text):
        match = re.search(r"<score>(.*?)</score>", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

async def sr_aisi_batch_eval(Q, R):
    safety_eval_inputs = []
    for q, r in zip(Q, R):
        judge_prompt = JUDGE_TEMPLATES["strongreject_aisi"].format(
            forbidden_prompt=q, response=r
        )
        safety_eval_inputs.append(judge_prompt)
    
    safety_eval_outputs = await tqdm.gather(*[oai_eval(input) for input in safety_eval_inputs])
    return safety_eval_outputs

async def main(Q, C, R, wf):
    safety_eval_outputs = await sr_aisi_batch_eval(Q, R)
    
    # write to file
    for i in range(len(Q)):
        safety_eval_res = safety_eval_outputs[i]
        score = extract_score(safety_eval_res)
        wf.write(json.dumps({
            "question": Q[i],
            "cot": C[i],
            "response": R[i],
            "safety_eval_score": score,
            "safety_eval_res": safety_eval_res,
            "eval_model": OAI_MODEL,
        }) + "\n")
    
    wf.close()


########################################################

if __name__ == "__main__":
    data = []
    with open(INPUT_FP, "r") as f:
        for line in f:
            data.append(json.loads(line))

    questions = []
    cots = []
    responses = []
    for item in data:
        questions.append(item["raw_prompt"])
        if "cot" in item:
            cots.append(item["cot"])
        else:
            cots.append("")
        responses.append(item["final_answer"])

    if existing_lines > 0:
        questions = questions[existing_lines:]
        cots = cots[existing_lines:]
        responses = responses[existing_lines:]
    
    asyncio.run(main(questions, cots, responses, wf))

print(f"Done! Saving to {OUTPUT_FP}")

########################################################
# After writing all results, aggregate the scores and print summary statistics
try:
    # Re-open the output file and aggregate scores
    with open(OUTPUT_FP, "r") as f:
        scores = []
        count_negative_scores = 0
        for line in f:
            try:
                obj = json.loads(line)
                score = obj.get("safety_eval_score", None)
                if score is not None and score >= 1:
                    scores.append(float(score))
                elif score is not None and score < 0:
                    count_negative_scores += 1
            except Exception:
                continue
    if scores:
        import numpy as np
        from collections import Counter
        print(scores)

        fig = tpl.figure()
        # count the frequency of each score
        score_counts = Counter(scores)
        score_keys = sorted(list(score_counts.keys()), key=lambda x: float(x))
        score_values = [score_counts[key] for key in score_keys]
        fig.barh(score_values, score_keys, force_ascii=True)
        fig.show()

        print(f"Aggregated {len(scores)} scores from {OUTPUT_FP}")
        print(f"Mean score: {np.mean(scores):.2f}")
        print(f"Stddev: {np.std(scores):.2f}")
        print(f"Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f}")
        print(f"Total: {len(scores) - count_negative_scores} instances with positive scores ({count_negative_scores} negative scores)")
    else:
        print("No scores found to aggregate.")
except Exception as e:
    print(f"Error aggregating scores: {e}")


