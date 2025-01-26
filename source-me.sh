#!/bin/bash

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${PWD}/env/lib64/python3.12/site-packages/nvidia/cublas/lib:${PWD}/env/lib64/python3.12/site-packages/nvidia/cudnn/lib
export LLAMA_CPP_LIB=${PWD}/env/lib/python3.12/site-packages/llama_cpp/lib/libllama.so

source env/bin/activate
