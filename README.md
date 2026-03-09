# AMD LLM Reproduce

This repository contains scripts, configurations, and experiment results for reproducing LLM (Large Language Model) workflows — **Pre-training**, **Fine-tuning**, and **Inference** — on AMD MI210 GPUs, based on the paper:

> ["Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models"](https://arxiv.org/abs/2311.03687)

---

## Repository Structure

```
AMD-LLM-Reproduce/
├── Pretrain/        # Pre-training experiments with DeepSpeed ZeRO strategies
├── Finetune/        # Fine-tuning experiments with LoRA and DeepSpeed
└── Inference/       # Inference benchmarks using vLLM and TGI
```

---

## Modules

### Pretrain

Reproduces pre-training performance using DeepSpeed.

Four strategies are benchmarked:
| Strategy | Throughput | Peak Memory |
|---|---|---|
| RQ (Quantization) | 1675.38 tokens/s | 9.23 GB |
| ZeRO-2 + Offload | 123.22 tokens/s | 15.06 GB |
| ZeRO-3 | 976.53 tokens/s | 40.34 GB |
| ZeRO-3 + Offload | 66.05 tokens/s | 4.58 GB |

- `configs/` — DeepSpeed configuration files (ZeRO-2/3 variants)
- `scripts/` — Launch scripts for normal and quantized training
- `run/` — Core Python scripts (`pretrain.py`, `quantize.py`, `download.py`, `utils.py`)
- `logs/` — Experiment log files
- `setup/` — Environment setup script for AMD platform (`amd_setup.sh`)

### Finetune

Fine-tunes LLMs using **LoRA (PEFT)** and **DeepSpeed** on AMD GPUs.

- `script/train.py` — Main training script using HuggingFace Transformers + TRL
- `script/ds_*.json` — DeepSpeed configs: naive, ZeRO-2, ZeRO-2 + Offload, ZeRO-3
- `data/result.csv` — Experiment results

### Inference

Benchmarks LLM inference throughput on a **single AMD MI210** using **vLLM** and **TGI (Text Generation Inference)** with Llama-3 8B.

| Framework | Throughput |
|---|---|
| vLLM | 371 tokens/s ± 12.4 |
| TGI | 216 tokens/s ± 8.28 |

- `scripts/` — Server/client launch scripts and benchmark scripts for both vLLM and TGI
- `experiments_data/` — Raw result CSVs (`vLLM_results.csv`, `TGI_results.csv`)

## Server Specification
| Part | Description | Numbers |
| :--- | :--- | :--- |
| CPU | AMD EPYC 9654 2.4GHz | 2 |
| GPU | AMD Instinct MI210 64GB | 3 |
| Memory | Micron DDR5 4800 64GB | 12 |
| Storage | Samsung PM9A3 4TB | 2 |
| Ethernet | Intel 82599ES 10GbE SFP+ <br> Intel I350 1GbE | 2 <br> 2 |
| OS | Proxmox Virtual Environment 8.0 | - |