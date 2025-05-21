set -x
uv pip install pylatexenc
uv pip install wandb
export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
MODEL_NAME=Qwen/Qwen2.5-7B
PROJECT_NAME=Qwen2.5-7B_Mix-Math-yt
EXPERIMENT_NAME=${MODEL_NAME}-rs-mstep
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}

python3 running_scripts/generation_to_hub.py \
    --datafiles ./data/mix-math-emodel-round2/generation.parquet \
    --hub Yuanxin-Liu/mix-math-7b-emodel-round2-rs

python3 -m verl.trainer.main_filter \
    --datafiles ./data/mix-math-emodel-round2/generation.parquet \
    --output_files ./data/mix-math-emodel-round2/generation-filtered.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/mix-math-emodel-round2/generation-filtered.parquet \
        data.val_files=./data/mix-math-emodel-round2/generation-filtered.parquet \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.train_batch_size=8 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL_NAME} \
        model.lora_rank=32 \
        model.lora_alpha=128 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.total_epochs=4 \
        trainer.logger=['console','wandb'] \
        optim.lr=1e-6 \
        ulysses_sequence_parallel_size=2 \
        use_remove_padding=true

for CHECKPOINT in ${SAVE_LOCAL_DIR}/global_step_*; do
    STEP=$(basename $CHECKPOINT)  # Extracts "global_step_X"
    HUB_MODEL_ID="Yuanxin-Liu/estep-rs-round2-epoch4-step-${STEP}"

    echo "Pushing checkpoint: $STEP to Hugging Face Hub at $HUB_MODEL_ID"

    python running_scripts/push_to_hub.py \
        --model_name_or_path ${MODEL_NAME} \
        --adapter_path ${CHECKPOINT} \
        --hub_model_id ${HUB_MODEL_ID}
done