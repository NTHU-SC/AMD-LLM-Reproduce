#!/bin/bash

# 模型與快取目錄（請依實際情況調整）
export MODEL_DIR=/home/tmouse/llm-inference/model/Llama-3-8B-Instruct
export HF_CACHE_DIR=~/llm-inference/cache/vllm-hub
mkdir -p "$HF_CACHE_DIR"

# vLLM ROCm 映像與容器設定
export VLLM_IMAGE="rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909"
export VLLM_CONTAINER_NAME="$USER-vllm_llama3_8b_server"
# TODO, 更改 port
# 注意不要搶 PORT!
export HOST_PORT=8000

echo "--- 準備啟動 vLLM (ROCm) 服務 ---"
echo "模型路徑: $MODEL_DIR"
echo "HF 快取:  $HF_CACHE_DIR"
echo "使用映像檔: $VLLM_IMAGE"
echo "服務將在 http://localhost:$HOST_PORT 上提供 (OpenAI 相容 API)"
echo "-----------------------------------"

sudo docker pull rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
sudo docker run \
	--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
	--device=/dev/kfd --device=/dev/dri --group-add video \
	--rm \
	-it \
	--name "$VLLM_CONTAINER_NAME" \
	--ipc=host --shm-size 64g \
	-p "$HOST_PORT":8000 \
	-v "$MODEL_DIR":/data:rw \
	-v "$HF_CACHE_DIR":/data-cache:rw \
	-e HUGGINGFACE_HUB_CACHE=/data-cache \
	"$VLLM_IMAGE" \
	python3 -m vllm.entrypoints.openai.api_server \
	--model /data \
	--host 0.0.0.0 \
	--port 8000 \
	--max-model-len 4096

# 使用方式：
# curl http://localhost:$HOST_PORT/v1/chat/completions \
# 	-H 'Content-Type: application/json' \
# 	-d '{
# 		"model": "/data",
# 		"messages": [{"role": "user", "content": "請用繁體中文介紹你自己!"}]
# 		}'

