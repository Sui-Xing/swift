# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "output/yi-6b-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
