#! /bin/bash

module purge
module load rocm

export CC=clang
export CXX=clang++

conda create --prefix /home/sky/pretrain-deepspeed python=3.11 -y
conda activate /home/sky/pretrain-deepspeed
pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install deepspeed-kernels
pip install flash-attn
pip install transformers datasets tokenizers
pip install numpy tqdm nltk
pip install accelerate

cd $HOME
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a" -S . 
make
pip install -e .

export CC=gcc
export CXX=g++
# export LD_LIBRARY_PATH=/home/sky/pretrain-deepspeed/lib:$LD_LIBRARY_PATH
# export LD_PRELOAD=/home/sky/opt/aocc-compiler-5.0.0/lib/libomp.so
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.17.4 --no-build-isolation