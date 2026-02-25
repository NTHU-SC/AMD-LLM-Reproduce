#!/bin/bash

# 使用本地模型：請確認目錄存在且內容完整
export MODEL_DIR="/home/tmouse/llm-inference/model/Llama-3-8B-Instruct"
export HF_CACHE_DIR=/home/tmouse/llm-inference/cache/tgi-hub
mkdir -p "$HF_CACHE_DIR"

# export TGI_IMAGE="ghcr.io/huggingface/text-generation-inference:latest-rocm"
# export TGI_IMAGE="ghcr.io/huggingface/text-generation-inference:3.3.6-rocm"
export TGI_IMAGE="ghcr.io/huggingface/text-generation-inference:2.3.0-rocm"
export TGI_CONTAINER_NAME="$USER-tgi_llama3_8b_server"
# TODO, 更改 port
# 注意不要搶 PORT!
export HOST_PORT=8080
echo "--- 準備啟動 TGI 服務 ---"
echo "本地模型路徑: $MODEL_DIR"
echo "使用映像檔: $TGI_IMAGE"
echo "服務將在 http://localhost:$HOST_PORT 上提供"
echo "--------------------------"

# 若仍需從 Hub 抓補檔，建議設定 HUGGINGFACE_HUB_TOKEN（可選）
[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && echo "[提示] 未設定 HUGGINGFACE_HUB_TOKEN（可選）。" >&2


# --- 執行 Docker 指令 ---
# 使用 docker run 來建立並啟動一個新的容器

sudo docker run \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --rm \
    -it \
    --name "$TGI_CONTAINER_NAME" \
    --ipc=host --shm-size 64g \
    -p "$HOST_PORT":80 \
    -v "$MODEL_DIR":/data:rw \
    -v "$HF_CACHE_DIR":/data-cache:rw \
    -e HUGGINGFACE_HUB_CACHE=/data-cache \
    "$TGI_IMAGE" \
    --model-id "/data" \
    --hostname 0.0.0.0 \
    --trust-remote-code \
    --max-total-tokens 4096 \
    --max-input-tokens 3072

# curl http://localhost:8080/generate \
#     -X POST \
#     -H "Content-Type: application/json" \
#     -d '{
#         "inputs": "What is Deep Learning?",
#         "parameters": {
#             "max_new_tokens": 256,
#             "do_sample": true,
#             "temperature": 0.7,
#             "top_p": 0.95
#         }
#     }'
