############ Projected compliance behavior (Qwen2.5) #############
OUTPUT_DIR=./scripts/mech_interp/proj_outputs # output directory
DATA_FP=./artifacts/s1.1-7B-think500-answer200-start0-end313.jsonl # inference jsonl file


# turn inference jsonl file into projected compliance csv file
python ./scripts/mech_interp/prep_proj_csv.py \
    --output_dir $OUTPUT_DIR \
    --fp $DATA_FP \
    --persona_type compliance

# compute projected compliance vector
BASE_FNAME=$(basename $DATA_FP .jsonl)
FP=$OUTPUT_DIR/${BASE_FNAME}_compliance.csv    

cd ./scripts/mech_interp/persona_vectors
PERSONA_VECTOR=./scripts/mech_interp/persona_vectors/persona_vectors/Qwen2.5-7B-Instruct/compliant_response_avg_diff.pt

CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection_modified \
    --file_path $FP \
    --vector_path $PERSONA_VECTOR \
    --layer_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type prompt_last_proj