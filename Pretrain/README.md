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

### Repository Structure and Execution

The repository is structured to manage scripts, configurations, and results:

* `scripts`: Contains the execution scripts. You must modify certain contents, such as directory paths and model paths, before execution.

* `configs`: Contains the DeepSpeed configuration files (.json) required to activate the four benchmarked methods ($Z2+O$, $Z3$, $Z3+O$, and $Q$).

* `logs`: The designated directory for storing experiment log files.

* `run`: The actual core Python script (pretrain_llm.py) that is executed when called by the wrapper scripts

## Results
- RQ: 1675.38 tokens/s, Peak memory usage 9.23 GB
- Z2+R+O: 123.22 tokens/s, Peak memory usage 15.06 GB
- Z3: 976.53 tokens/s, Peak memory usage 40.34 GB
- Z3+R+O: 66.05 tokens/s, Peak memory usage 4.58 GB