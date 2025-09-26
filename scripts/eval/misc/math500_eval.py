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
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file containing model outputs")
args = parser.parse_args()
INPUT_FP = pathlib.Path(args.input_file)

from openai import OpenAI
client = openai.AsyncOpenAI()
OAI_MODEL = "gpt-5"

OUTPUT_DIR = INPUT_FP.parent / "math500_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FP = OUTPUT_DIR / f"math500-gpt5-labeled_{INPUT_FP.stem}.jsonl"


if OUTPUT_FP.exists():
    logger.warning(f"{OUTPUT_FP} exists, will overwrite it.")

wf = open(OUTPUT_FP, "w+", encoding="utf-8", buffering=1)
logger.info(f"Saving to {OUTPUT_FP}")


########################################################
# adapted from https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/eval/lm-evaluation-harness/lm_eval/tasks/openai_math/utils.py
JUDGE_TEMPLATE_MATH500 = """
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['12']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

1

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


async def oai_eval(input_text):
    try:
        response = await client.responses.create(
            model=OAI_MODEL,
            input=input_text,
        )
    except Exception as e:
        print(f"Error evaluating input: {e}")
        return "-1"
    return response.output_text

async def math500_batch_eval(gold_answers, predicted_answers):
    math500_eval_inputs = []
    for gold, pred in zip(gold_answers, predicted_answers):
        pred = f"[{pred}]"
        prompt = JUDGE_TEMPLATE_MATH500 % {"expression1": gold, "expression2": pred}
        math500_eval_inputs.append(prompt)
    
    math500_eval_outputs = await tqdm.gather(*[oai_eval(input) for input in math500_eval_inputs])
    return math500_eval_outputs

async def main(gold_answers, predicted_answers, full_generations, wf):
    math500_eval_outputs = await math500_batch_eval(gold_answers, predicted_answers)
    correct = math500_eval_outputs.count("1")
    accuracy = correct / len(math500_eval_outputs) * 100
    logger.info(f"Accuracy: {accuracy:.2f}%")
    
    for i in range(len(gold_answers)):
        oai_answer = math500_eval_outputs[i]
        wf.write(json.dumps({
            "full_generations": full_generations[i],
            "gold_answer": gold_answers[i],
            "predicted_answer": predicted_answers[i],
            "oai_answer": oai_answer,

        }) + "\n")
    
    wf.write(json.dumps({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(math500_eval_outputs),
        "eval_model": OAI_MODEL,
    }) + "\n")
    
    wf.close()

def extract_boxed_answer(text):
    # extract from strings inside the \boxed like "\boxed{(3, \dfrac{\pi}{2})}"
    match = re.search(r"\\boxed\{(.*)\}", text)
    if match:
        return match.group(1)
    return None
########################################################

if __name__ == "__main__":
    data = []
    with open(INPUT_FP, "r") as f:
        for line in f:
            data.append(json.loads(line))

    gold_answers = []
    full_generations = []
    predicted_answers = []
    for item in data:
        gold_answers.append(item["answer"])
        full_generations.append(item["full_conversation"])

        # extract the answer from the full generations
        _generations = item["full_conversation"].split("<|im_start|>answer\nFinal Answer:", 1)[-1].strip()
        extracted_answer = extract_boxed_answer(_generations)
        if extracted_answer is None:
            logger.warning("No boxed answer found in generations:", _generations)
            predicted_answers.append("None")
        else: 
            predicted_answers.append(extracted_answer)

    asyncio.run(main(gold_answers, predicted_answers, full_generations, wf))

logger.success(f"Done! Saving to {OUTPUT_FP}")
