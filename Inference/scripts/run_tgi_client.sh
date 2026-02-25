#!/bin/bash

cd /home/tmouse/llm-inference/HPC-II_LLM-Reproduce-Inference
source hpc-II-llm/bin/activate

export MY_COMPUTE_NODE=localhost

PROMPTS_N=100 LL_HOST=$MY_COMPUTE_NODE LL_PORT=8080 python run_tgi.py

# curl -sS http://localhost:$LL_PORT/generate \
# 		-H 'Content-Type: application/json' \
# 		-d '{
# 			"inputs": "請用繁體中文介紹你自己！",
# 			"parameters": {
# 				"max_new_tokens": 256,
# 				"temperature": 0.7,
# 				"top_p": 0.95
# 			},
# 			"details": true
# 		}'