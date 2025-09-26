# Self-Jailbreaking of Reasoning Language Models

This is the official repository for the paper "Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training".

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


## 🗂️ Repo Structure
```text
./                      # root repo 
|___examples/           # example of self-jailbreaking (model outputs)
|__scripts/
    |__eval             # safety evaluation (strongreject)
    |__inference        # model generations
    |__interp           # mech-interp experiments
    |__selfjb_detect    # self-jailbreaking detection
    |__safety_training  # safety reasoning training
```


## 🏃‍♂️ Safety Reasoning Training: Safe-s1.1-7B

You can set `--sample_size` hyperparameter to decide the percentage of training samples to use from the [STAR-1](https://huggingface.co/datasets/UCSC-VLAA/STAR-1) safety reasoning dataset.

```python3
# train with 5% of STAR-1 safety dataset
# model will be saved to $OUT directory

python3 ./scripts/safety_training/finetune_qwen_safe.py
    --sample_size 0.05 \ 
    --output_dir $OUT
```
