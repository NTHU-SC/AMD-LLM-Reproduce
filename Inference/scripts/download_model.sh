#!/bin/bash

python3 -m venv hpc-II-llm
source hpc-II-llm/bin/activate

pip install huggingface-hub

hf auth login
# Add your Hugging Face token when prompted

hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./model/Llama-3-8B-Instruct

hf download meta-llama/Llama-2-7b-hf --local-dir ./model/Llama-2-7B-hf

# NousResearch/Llama-2-7b-chat-hf

hf download meta-llama/Llama-2-7b-chat-hf --local-dir ./model/Llama-2-7B-Chat