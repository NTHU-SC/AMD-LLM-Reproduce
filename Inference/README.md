# Inference

## Goal
Run Llama-3 8B inference on single MI210 with vLLM and TGI(Text Generation Inference).

## Results
After removing outliers, vLLM results are presented 371 token/s ± 12.4.
The average throughput for TGI was 216 token/s ± 8.28.

## Software Libraries and Framework Versions

### Inference Frameworks (Docker Images)

| Framework | Docker Image | Version |
|-----------|-------------|---------|
| vLLM | `rocm/vllm` | `rocm6.4.1_vllm_0.10.1_20250909` |
| TGI (Text Generation Inference) | `ghcr.io/huggingface/text-generation-inference` | `2.3.0-rocm` |

### Runtime Configuration

| Parameter | vLLM | TGI |
|-----------|------|-----|
| Max model length / total tokens | 4096 | 4096 |
| Max input tokens | — | 3072 |
| Precision | FP16 | FP16 |
| Quantization | None | None |


## Discussion
### Framework Hierarchy: Similarities and Differences

Our reproduction results reveal a significant contrast in framework hierarchy compared to the paper's findings, highlighting the strong impact of hardware generation on inference performance.

**The "LightLLM" Reversal:**
In the paper (Fig. 6 and Section VI.A), LightLLM significantly outperforms both vLLM and TGI on the NVIDIA A800 platform. However, in our experiment conducted on AMD MI210, LightLLM was not available because software compatibility. This contradicts the paper's framework hierarchy observed on high-end GPUs, and confirms the hypothesis that LightLLM is heavily optimized for newer GPU architectures (Ampere+). Our MI210 results similarly show that not all frameworks are equally portable across hardware vendors.

**vLLM's Consistency:**
The paper demonstrates that vLLM maintains stable and competitive performance across different platforms. Our results align with this observation — vLLM achieved the highest throughput on MI210 (~371 tok/s), outperforming TGI (~216 tok/s) even without the hardware-specific optimizations (e.g., FlashAttention-2) that benefit Ampere-based GPUs.

### Discrepancies and Possible Reasons

Our measured throughput for vLLM on MI210 (~371 tok/s) is considerably lower than the paper's results on A800 (>5000 tok/s). Beyond the raw compute difference between GPU generations, three specific technical factors contribute to the discrepancy:

1. **Hardware Architecture Features (CDNA2 vs. Ampere):**
   The paper explicitly notes that LightLLM is *"specifically optimized for high-performance GPUs such as the A800/A100 series"*. AMD MI210 (CDNA2) uses the ROCm software stack, and not all CUDA-optimized paths — including FlashAttention-2 and specific Triton kernels relied upon in the paper — translate directly to the ROCm ecosystem. This limits the achievable throughput compared to native Ampere (A800) results.

2. **Dataset Composition:**
   The paper utilized a synthetic dataset with a fixed input length of 512 tokens and generated bursts of concurrent requests to stress-test throughput. In contrast, our experiment used `prompts.json` with varying real-world prompt lengths. Since inference throughput is highly sensitive to the prefill/decode ratio, this workload difference shifts the compute-bound vs. memory-bound profile of the benchmark, making direct numerical comparisons difficult.

3. **Software Stack & Quantization:**
   The paper applied optimizations such as 4-bit quantization and compared across multiple kernel versions. Our reproduction used the base Llama-3 8B model strictly in FP16 precision without quantization. The absence of quantization increases memory bandwidth pressure per token, leading to lower measured throughput compared to the optimized configurations reported in the literature.
