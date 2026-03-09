# Finetune
## Introduction
This repository contains the scripts and configurations required to reproduce the fine-tuning performance experiment from the paper: *"Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models".*

## Software Libraries and Framework Versions
- ROCm: 6.4
- Python: 3.9.25
- PyTorch stack (with ROCm6.4 support):
    - torch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0, pytorch-triton-rocm 3.4.0
- Fine-tuning frameworks:
    - transformers 4.57.3, datasets 4.4.2, accelerate 1.10.1, peft 0.17.1, trl 0.24.0, deepspeed 0.18.3, xformers 0.0.32.post2
- Quantization / optimization:
    - bitsandbytes 0.48.2, numpy 2.0.2, scipy 1.13.1, safetensors 0.7.0, einops 0.8.1
- Utilities:
    - huggingface-hub 0.36.0, tokenizers 0.22.1, tqdm 4.67.1, pandas 2.3.3, pyarrow 21.0.0, regex 2025.11.3

## Results
Our reproduction results on AMD MI210 GPUs are summarized below:
| Method                     | Mi210 Throughputs (token/s) | Mi210 Peak Memory (GB) |
| -------------------------- | --------------------------- | ---------------------- |
| LoRA                       | 1020.01                     | 17.49                  |
| LoRA + ZeRO-2              | 1120.19                     | 19.22                  |
| LoRA + ZeRO-2 + Offloading | 710.715                     | 17.44                  |
| QLoRA                      | 738.845                     | 8.475                  |

## Discussion

### QLoRA vs. LoRA on MI210

**Measurements**

- Throughput: LoRA ≈ 1020 tokens/s, QLoRA ≈ 740 tokens/s (≈ 25% lower).  
- Peak memory: LoRA ≈ 17.5 GB, QLoRA ≈ 8.5 GB (≈ 2× saving).

**Explanation**

- This follows Table IX in the original paper: LoRA delivers the highest fine‑tuning throughput, while QLoRA sacrifices speed for a much smaller memory footprint.  
- QLoRA stores base weights in 4‑bit NF4 and must dequantize them to FP16/BF16 on every forward and backward pass. The extra dequantization work and memory traffic directly reduce effective throughput on MI210.  
- Overall, MI210 shows the same strategy ordering as the paper: **LoRA > QLoRA** in throughput, with QLoRA being suitable mainly when GPU memory is the primary constraint.

### ZeRO‑2 and Offloading on MI210

**Ｍeasurements**

- LoRA: ≈ 1020 tokens/s.  
- LoRA + ZeRO‑2: ≈ 1120 tokens/s (slightly faster than LoRA).  
- LoRA + ZeRO‑2 + Offloading: ≈ 710 tokens/s (about 1/3 slower than LoRA).

**Explanation**

- The trend is consistent with the paper: ZeRO‑2 is beneficial, while offloading is the slowest configuration because of CPU–GPU communication overhead.  
- In our single‑GPU MI210 setup, ZeRO‑2 cannot use cross‑GPU sharding. The small throughput gain likely comes from ROCm + DeepSpeed managing optimizer states and memory layout more efficiently than the plain PyTorch baseline.  
- Offloading moves relatively small LoRA optimizer states to CPU, which saves little HBM but forces frequent PCIe transfers. Given PCIe’s much higher latency and lower bandwidth than on‑GPU HBM2e, communication cost dominates and throughput drops sharply.  
- As in the paper, ZeRO‑Offload only makes sense under extreme memory pressure; for 7B LoRA on MI210 it is clearly not a good trade‑off.


### MI210 vs. Original Paper Platforms (A800 / RTX 30/40)

**Measurements**

- MI210 per‑GPU throughput across all methods: roughly 700–1100 tokens/s. 
- This is far below the per‑GPU throughput implied by A800 in the paper (several thousand tokens/s for similar LoRA‑style fine‑tuning), but roughly comparable to or slightly higher than the reported RTX 30/40 results.

**Explanation**

- **Hardware:** A800 (Ampere) offers BF16‑optimized Tensor Cores and about 2 TB/s HBM2e bandwidth, while MI210 (CDNA2) provides about 1.6 TB/s and lower peak BF16/FP16 compute. Higher throughput on A800 is therefore expected for transformer fine‑tuning workloads.  
- **Software:** The paper uses a mature CUDA‑based stack (cuBLAS, cuDNN, NCCL, fused kernels). Our MI210 experiments run on ROCm with PyTorch 2.8 and DeepSpeed 0.18.3, where some kernels and communication paths are less optimized than their CUDA counterparts.  
- **Positioning of MI210:** Compared to RTX 30/40 GPUs, MI210 benefits from HBM2e’s higher bandwidth and capacity relative to GDDR6X, which aligns with the paper’s observation that these workloads are often memory‑bound. This explains why MI210 can roughly match or slightly exceed the per‑GPU throughput of consumer GPUs, even though it still falls short of A800‑class performance.

***

### Overall Conclusion

- The relative behavior of all methods on MI210 closely matches the original paper: LoRA has the highest throughput, QLoRA trades speed for memory, ZeRO‑2 is helpful, and offloading is consistently the slowest.  
- The absolute performance of MI210 is lower than A800 but roughly in the same range as RTX 30/40 GPUs, which is consistent with the differences in hardware capability (compute and HBM bandwidth) and software stack maturity (ROCm vs. CUDA).  
- Taken together, our results confirm that the paper’s main conclusions about fine‑tuning strategy trade‑offs generalize to AMD MI210, while also highlighting the performance gap between MI210 and newer NVIDIA data‑center GPUs such as A800.
