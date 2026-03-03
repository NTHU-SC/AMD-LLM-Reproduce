import argparse
import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time
import nltk
import os

# NLTK words
nltk.download('words')
from nltk.corpus import words
word_list = words.words()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import BenchmarkTimer

# -----------------------------
# Random text generation utils
# -----------------------------
def generate_random_text(tokenizer, target_token_length=350, max_trials=10):
    for _ in range(max_trials):
        approx_word_len = int(target_token_length * 0.75)
        sent_words = random.choices(word_list, k=approx_word_len)
        text = ' '.join(sent_words)
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=target_token_length,
            return_tensors="pt"
        )
        if tokens.input_ids.size(1) == target_token_length:
            return tokens.input_ids.squeeze(0)
    return tokens.input_ids.squeeze(0)

def generate_batch(tokenizer, batch_size, seq_len, device):
    batch = torch.stack([generate_random_text(tokenizer, seq_len) for _ in range(batch_size)])
    return batch.to(device)

# -----------------------------
# Training step
# -----------------------------
def train_step(ds_engine, inputs, labels):
    outputs = ds_engine(inputs, labels=labels)
    loss = outputs.loss
    ds_engine.backward(loss)
    ds_engine.step()
    return loss.item()

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--model_name', type=str, default='/home/sky/models/Llama-2-7b-hf')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=350)
    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Distributed init
    # -----------------------------
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # -----------------------------
    # Model (no quantization)
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": device},
        torch_dtype=torch.float16,
        use_cache = False
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    # -----------------------------
    # DeepSpeed initialize
    # -----------------------------
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
        dist_init_required=False
    )

    model.train()

    # -----------------------------
    # Training loop
    # -----------------------------
    timer = BenchmarkTimer(warm_up_steps=30, total_steps=args.total_steps)

    for step in range(args.total_steps + 30):
        inputs = generate_batch(tokenizer, args.batch_size, args.seq_len, ds_engine.device)
        labels = inputs.clone()

        ts = timer.step_start()

        loss_val = train_step(ds_engine, inputs, labels)

        local_tokens = torch.tensor([args.batch_size * args.seq_len], device=ds_engine.device)
        if dist.is_initialized() and ds_engine.world_size > 1:
            dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM)
        
        step_time, status = timer.step_end(ts, step, local_tokens.item())

        if ds_engine.local_rank == 0:
            tps = local_tokens.item() / step_time
            print(f"[{status}] Step {step+1}: Loss {loss_val:.4f} | {tps:.2f} tokens/s")

    if ds_engine.local_rank == 0:
        timer.print_final_stats(0)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()