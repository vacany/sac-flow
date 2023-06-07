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


#python pipeline/sceneflow.py \
#      --exp_name forward_flow-ours \
#      --use_smoothness 1 \
#      --use_visibility_smoothness 1 \
#      --use_reverse_nn 0 \
#      --use_forward_flow_smoothness 1 \
#      --iters 10
