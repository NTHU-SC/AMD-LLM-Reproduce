#!/bin/bash

cd /home/tmouse/llm-inference/HPC-II_LLM-Reproduce-Inference
source hpc-II-llm/bin/activate

PROMPTS_N=100 VLLM_HOST=localhost VLLM_PORT=8000 python3 run_vllm.py

# curl http://localhost:$HOST_PORT/v1/chat/completions \
# 	-H 'Content-Type: application/json' \
# 	-d '{
# 		"model": "/data",
# 		"messages": [{"role": "user", "content": "請用繁體中文介紹你自己!"}]
# 		}'