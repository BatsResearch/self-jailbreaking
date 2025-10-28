cd ./scripts/mech_interp/persona_vectors

COEF="[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]"
LAYER="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]"
OUT_FP="eval_persona_eval/steering_coef=${COEF}_layer=${LAYER}.csv"
if [ ${#OUT_FP} -gt 150 ]; then
    SHORT_FP="eval_persona_eval/steering_short_$COEF.csv"
    echo "OUT_FP too long, using $SHORT_FP instead"
    OUT_FP=$SHORT_FP
fi

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona_steer_multiple \
    --model simplescaling/s1.1-7B \
    --trait eval_strongreject \
    --output_path $OUT_FP \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval \
    --steering_type response \
    --coef $COEF \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/harmfulness_response_avg_diff.pt \
    --layer $LAYER \
    --n_per_question 1 \
    --max_tokens 100 \
    --already_prefilled \
    --skip_judge


### output evaluation results
python3 ../../../scripts/eval/strongreject_eval_csv.py \
    --input_file ${OUT_FP}