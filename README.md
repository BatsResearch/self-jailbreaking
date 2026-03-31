# Self-Jailbreaking of Reasoning Language Models

This is the official repository for the paper "[Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training](https://arxiv.org/abs/2510.20956)".


## 🚀 Quick Setup

```sh
# set your OPENAI API KEY
export OPENAI_API_KEY=... 

# set up your virtual environment
uv venv --python 3.10
source .venv/bin/activate

# install packages
uv pip install -r requirements.txt
```


## 🗂️ Structure
```text
./                      # root repo 
|___examples/           # example of self-jailbreaking (model outputs)
|__scripts/
    |__eval/                # safety evaluation (strongreject)
    |__inference/           # model generations
    |__interp/              # mech-interp experiments
    |__selfjb_detect/       # self-jailbreaking detection
    |__safety_training/     # safety reasoning training
```

## ⛓️‍💥 Identifying Self-Jailbreaking within CoT

| Scripts    | Purpose |
| -------- | ------- |
| scripts/inference/vllm_gen{_s1}.py    | Inference on StrongReject prompts |
| scripts/eval/strongreject_eval.py  | Evaluation of Attack Success Rate (ASR) |
| scripts/selfjb_detect/selfjb_detect.py | Detection of Self-Jailbreaking Occurences     |
| scripts/selfjb_detect/selfjb_stats.py | Stats of Self-Jailbreaking Occurences     |

Here's an example of the usage of the scripts to annotate on self-jailbreaking instances for the s1.1-7B model.

```bash
python3 ./scripts/inference/vllm_gen_s1.py --model_name "simplescaling/s1.1-7B"
python3 ./scripts/eval/strongreject_eval.py --input_file ./outputs/s1.1-7B-think500-answer200.jsonl
python3 ./scripts/selfjb_detect/selfjb_detect.py --input_file ./outputs/oai_aisi_eval/oai_aisi-gpt5-labeled_s1.1-7B-think500-answer200.jsonl
python3 ./scripts/selfjb_detect/selfjb_stats.py --input_file ./outputs/selfjb_detect/selfjb_detect-oai_aisi-gpt5-labeled_s1.1-7B-think500-answer200-start0-end313.jsonl


# NOTE: use `vllm_gen.py` for non-s1 models
# python3 ./scripts/inference/vllm_gen.py --model_name "microsoft/Phi-4-mini-reasoning" (for non-s1 models)
```

### 💾 Artifacts

`./artifacts` folder stores the artifacts of our model generations and self-jailbreaking annotations.

## 🔎 Mechanistic Analysis of Self-Jailbreaking CoT

Our work adapts Anthropic's [Persona Vector](https://github.com/safety-research/persona_vectors) repository by extending their steering to support multiple layers and correcting the normalization to be global norm. Follow the installation in `./scripts/mech_interp/persona_vectors` folder for usage.

Here are the scripts for reproducing our results.
| Scripts    | Purpose |
| -------- | ------- |
| scripts/mech_interp/compute_projection_vector.sh    | Obtain compliance and perceived harmfulness vectors |
| scripts/mech_interp/proj_compliance.sh | Obtaining compliance projection scores (Figure 4)     |
| scripts/mech_interp/s1_vect_graph(_viz).sh  | Plotting (and visualizing) the perceived harmfulness and compliance scores for Figure 6 |
| scripts/mech_interp/steering.sh | Steering on multiple layers     |

**💾 Artifacts**: For reproducibility, we provide our projection vectors in `./scripts/mech_interp/persona_vectors/persona_vectors/Qwen2.5-7B-Instruct`.

## 🏃‍♂️ Safety Reasoning Training: Safe-s1.1-7B

You can set `--sample_size` hyperparameter to decide the percentage of training samples to use from the [STAR-1](https://huggingface.co/datasets/UCSC-VLAA/STAR-1) safety reasoning dataset.

```bash
# train with 5% of STAR-1 safety dataset
# model will be saved to $OUT directory

python3 ./scripts/safety_training/train_safe_s1.py
    --sample_size 0.05 \ 
    --output_dir $OUT
```

🤗 We also uploaded our model weights to our [HuggingFace collection](https://huggingface.co/collections/BatsResearch/safe-s11-68d6577961edca25c9619470).


## Citation
```
@inproceedings{
yong2026selfjailbreaking,
title={Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training},
author={Zheng-Xin Yong and Stephen H. Bach},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=akbtPEZnDZ}
}
```
