#!/bin/bash

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${PWD}/env/lib64/python3.12/site-packages/nvidia/cublas/lib:${PWD}/env/lib64/python3.12/site-packages/nvidia/cudnn/lib

source env/bin/activate
