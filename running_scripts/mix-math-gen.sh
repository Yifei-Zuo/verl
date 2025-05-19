set -x

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}
export HF_TOKEN=
MODEL_NAME=Yuanxin-Liu/Qwen2.5-7B-e-step-round-2

python3 data_preprocess/math_r1_dataset.py
python3 data_preprocess/still_30k.py
python3 data_preprocess/aime_train_dataset.py

if [ -d "./data/mix-math" ]; then
    rm -rf ./data/mix-math
fi
python3 data_preprocess/create_math_data_mix.py --local_dir data/mix-math/train.parquet


python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=./data/mix-math/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=2 \
    data.output_path=./data/mix-math-emodel-round2/generation.parquet \
    model.path=$MODEL_NAME \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=262144

python3 running_scripts/generation_to_hub.py --push \
    --datafiles ./data/mix-math-emodel-round2/generation.parquet \
    --hub Yuanxin-Liu/mix-math-7b-emodel-round2-rs