### Creating Safe-s1.1-7B

Simply launch the script of `finetune_qwen_safe.py` to train the Safe-s1.1-7B model that is capable of safety reasoning.

```python3
# train with 5% of STAR-1 safety dataset
# model will be saved to $OUT directory

python3 /scripts/safety_training/finetune_qwen_safe.py
    --sample_size 0.05 \ 
    --output_dir $OUT
```