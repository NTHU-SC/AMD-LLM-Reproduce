# Pretrain

## Introduction

This repository contains the scripts and configurations required to reproduce the pre-training performance experiment from the paper: *"Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models"*.

**Full Specification:** Please refer to the [HPC-II Detailed Experiment Specification](https://hackmd.io/wkF3UrSLRaWHIJ8VB98aDg) for full experimental details, parameters, and analytical questions.

---

## Cluster Information

* **Hardware Specification:** Mi210 $\times$ 2
* **Model Repository Path:** `/home/sky/models`

---

## Environment Setup

The necessary dependency packages have been **pre-built**. Use the following commands to initialize your environment on the respective clusters:


### AMD Platform (Mi210)

```
source /etc/profile.d/modules.sh
module use /home/sky/modulefiles
module purge
module load rocm ucx ucc openmpi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sky/pretrain-deepspeed
```

---

## Software Libraries and Framework Versions

The following versions are used in our reproduction experiments:

### System Libraries
* **ROCm:** 6.4
* **UCX:** 1.18.1
* **UCC:** 1.4.4
* **OpenMPI:** 5.0.8

### Python Environment
* **Python:** 3.11
* **PyTorch:** 2.x with ROCm 6.4 support (`torch`, `torchvision` from `https://download.pytorch.org/whl/rocm6.4`)

### Deep Learning Frameworks
* **DeepSpeed:** 0.17.4 (built with `DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1`)
* **Transformers:** 4.57.1
* **Datasets:** 4.3.0
* **Tokenizers:** 0.22.1
* **Accelerate:** 1.11.0

### Optimization Libraries
* **BitsAndBytes:** Custom ROCm build from source (`COMPUTE_BACKEND=hip`, `BNB_ROCM_ARCH="gfx90a"`)
* **DeepSpeed-Kernels:** 0.0.1
* **Flash-Attention:** 2.8.3

### Utilities
* **NLTK:** For text generation
* **NumPy:** Numerical computing
* **tqdm:** Progress bars

**Note:** The detailed environment setup script is available in [`setup/amd_setup.sh`](setup/amd_setup.sh).

---

### Repository Structure and Execution

The repository is structured to manage scripts, configurations, and results:

* `scripts`: Contains the execution scripts. You must modify certain contents, such as directory paths and model paths, before execution.

* `configs`: Contains the DeepSpeed configuration files (.json) required to activate the four benchmarked methods ($Z2+O$, $Z3$, $Z3+O$, and $Q$).

* `logs`: The designated directory for storing experiment log files.

* `run`: The actual core Python script (pretrain_llm.py) that is executed when called by the wrapper scripts

## Results

Our reproduction results on AMD MI210 GPUs are summarized below:

| Strategy | Throughput (tokens/s) | Peak Memory (GB) |
|----------|----------------------|------------------|
| RQ (Quantization) | 1675.38 | 9.23 |
| Z2+R+O (ZeRO-2 + Offload) | 123.22 | 15.06 |
| Z3 (ZeRO-3) | 976.53 | 40.34 |
| Z3+R+O (ZeRO-3 + Offload) | 66.05 | 4.58 |

## Discussion

### Strategy Performance Hierarchy: Similarities and Differences

Our reproduction results on AMD MI210 reveal both consistent patterns and notable differences compared to the paper's findings on NVIDIA A800 GPUs, reflecting the impact of hardware architecture and software stack compatibility on pre-training performance.

**Quantization's Dominance:**
Consistent with the paper's findings, our RQ (Runtime Quantization) configuration achieved the highest throughput (~1675 tokens/s) among all tested strategies. The paper similarly demonstrates that quantization-based approaches can significantly reduce memory bandwidth pressure while maintaining competitive training throughput. This validates that quantization remains an effective optimization across different GPU architectures, though the absolute performance scales with the underlying hardware capability.

**ZeRO Strategy Hierarchy:**
The relative performance ordering of ZeRO strategies aligns with the paper's observations: ZeRO-3 without offloading (976.53 tokens/s) substantially outperforms both offloading configurations (Z2+R+O: 123.22 tokens/s, Z3+R+O: 66.05 tokens/s). This is expected, as CPU offloading introduces significant communication overhead between host and device memory. The paper's experiments on A800 show a similar pattern where offloading trades throughput for reduced GPU memory consumption.

**Memory-Throughput Trade-offs:**
Our results confirm the fundamental memory-throughput trade-off described in the paper: Z3+R+O achieves the lowest memory footprint (4.58 GB) but also the lowest throughput (66.05 tokens/s), while Z3 without offloading consumes the most memory (40.34 GB) but delivers higher throughput. This pattern is consistent across both AMD and NVIDIA platforms, indicating that the trade-off is inherent to the optimization strategy rather than hardware-specific.

### Discrepancies and Possible Reasons

While our AMD MI210 results maintain similar relative performance patterns to the paper's A800 results, the absolute throughput values are considerably lower. Several technical factors contribute to these discrepancies:

1. **Hardware Generation Gap (CDNA2 vs. Ampere):**
   The NVIDIA A800 (Ampere architecture) features higher peak FP16 throughput (~312 TFLOPS) compared to AMD MI210 (CDNA2, ~181 TFLOPS for FP16). This raw compute difference directly impacts training throughput, particularly for compute-bound workloads like transformer pre-training. Additionally, the A800's tensor cores provide specialized acceleration for matrix operations that may not have direct equivalents in MI210's matrix cores.

2. **Software Stack Maturity (ROCm vs. CUDA):**
   The paper's experiments leverage highly optimized CUDA kernels and libraries (cuBLAS, cuDNN, NCCL) that have been refined over many GPU generations. While ROCm has made significant progress, certain DeepSpeed optimizations and kernel fusions may not be as mature or may require different tuning parameters on AMD hardware. For instance, communication primitives in NCCL (used in the paper) versus RCCL (used in our reproduction) may exhibit different performance characteristics for collective operations like all-reduce and all-gather.

3. **Memory Bandwidth and Topology:**
   Although both platforms feature high-bandwidth memory (HBM2e), the memory subsystem architectures differ. The A800 (80 GB HBM2e) provides ~2 TB/s bandwidth, while MI210 (64 GB HBM2e) offers ~1.6 TB/s. This bandwidth difference particularly impacts offloading strategies (Z2+R+O, Z3+R+O), which rely heavily on PCIe transfers between GPU and CPU memory. The paper's higher absolute throughput for offloading configurations may reflect both faster GPU-side computation and more optimized host-device communication in the CUDA ecosystem.

4. **Quantization Implementation:**
   Our RQ configuration uses BitsAndBytes with ROCm support for 8-bit quantization. The paper may have employed different quantization libraries or precision formats (e.g., INT8 vs. FP8) that could benefit from vendor-specific acceleration. NVIDIA's recent support for FP8 on Ampere+ GPUs, if used in the paper, would provide additional performance advantages not available on MI210.

5. **Compiler and Runtime Optimizations:**
   The PyTorch and DeepSpeed stack on CUDA benefits from NVIDIA's proprietary compiler optimizations (NVCC, NVRTC) and JIT compilation paths. Our ROCm-based stack uses HIP and ROCm compilers, which may generate less optimized code for certain operations, particularly for dynamically generated kernels in DeepSpeed's runtime.

Despite these absolute performance differences, the qualitative insights from our reproduction remain valuable: the relative performance ordering of strategies, memory consumption patterns, and trade-off characteristics are preserved across platforms, confirming that the paper's strategic findings generalize beyond NVIDIA-specific hardware.