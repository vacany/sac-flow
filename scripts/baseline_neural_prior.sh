#!/bin/bash

# shellcheck disable=SC2164
cd $HOME/pcflow

GPU=2
# probably split to paralelize

python pipeline/sceneflow.py \
      --dataset argoverse \
      --exp_name neural_prior \
      --lr 0.008 \
      --gpu $GPU \
      \
      --l_ch 1 \
      --l_dt 0 \
      --l_sm 0 \
      --l_ff 0 \
      --l_vsm 0 \
      \
      --l_ch_bothways 1 \
      --smooth_K 0 \
      --KNN_max_radius 1 \
      --pc2_smoothness 0

# different args for 8192 and full point cloud ...

# https://arxiv.org/pdf/2304.09121.pdf supplementary material has results of flowstep, PV-Raft, PointPWC-Net
