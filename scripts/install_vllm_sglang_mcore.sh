#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32
export UV_LINK_MODE=copy
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip

echo "1. install inference frameworks and pytorch they need"
uv pip install --no-cache-dir "vllm==0.8.5" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata


echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    # pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn --no-build-isolation

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
# wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    # pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


# if [ $USE_MEGATRON -eq 1 ]; then
#     echo "4. install TransformerEngine and Megatron"
#     echo "Notice that TransformerEngine installation can take very long time, please be patient"
#     NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
#     pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
# fi


echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


# if [ $USE_MEGATRON -eq 1 ]; then
#     echo "6. Install cudnn python package (avoid being overrided)"
#     pip install nvidia-cudnn-cu12==9.8.0.87
# fi

uv pip install hydra-core datasets accelerate setuptools codetiming peft
uv pip uninstall wandb
uv pip install wandb --no-cache-dir

echo "Successfully installed all packages"
