cd ./scripts/mech_interp/persona_vectors

# NOTE: replace `compliant` with `harmfulness` to extract perceived harmfulness vectors


###### extract outputs
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait compliant \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/compliant_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name judge \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait compliant \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/compliant_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name judge \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract


##### generate vectors
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/compliant_pos_instruct_compliant_coherence.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/compliant_neg_instruct_compliant_coherence.csv \
    --trait compliant \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/