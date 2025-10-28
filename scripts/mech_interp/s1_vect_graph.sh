python3 ./scripts/mech_interp/s1_vect_graph.py \
    --model_name "simplescaling/s1.1-7B" \
    --harmfulness_direction ./scripts/mech_interp/persona_vectors/persona_vectors/Qwen2.5-7B-Instruct/harmfulness_response_avg_diff.pt \
    --compliance_direction ./scripts/mech_interp/persona_vectors/persona_vectors/Qwen2.5-7B-Instruct/compliant_response_avg_diff.pt \
    --output_dir /users/zyong2/data/zyong2/selfjb/gh/batsresearch/scripts/mech_interp/output