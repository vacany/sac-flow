#!/bin/bash

# shellcheck disable=SC2164
cd $HOME/pcflow

python pipeline/sceneflow.py \
      --exp_name nsfp \
      --use_smoothness 0 \
      --use_visibility_smoothness 0 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 0 \
      --iters 50 \
      --max_radius 80

# different args for 8192 and full point cloud ...

# https://arxiv.org/pdf/2304.09121.pdf supplementary material has results of flowstep, PV-Raft, PointPWC-Net
