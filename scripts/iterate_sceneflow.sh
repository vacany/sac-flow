#!/bin/bash

cd $HOME/pcflow

# TODO create csv file to parse lines to bash, something like https://www.howtogeek.com/826549/parse-csv-data-in-bash/

python pipeline/sceneflow.py \
      --exp_name baseline5000 \
      --use_smoothness 0 \
      --use_visibility_smoothness 0 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 0 \
      --iters 5000

python pipeline/sceneflow.py \
      --exp_name smoothness5000 \
      --use_smoothness 1 \
      --use_visibility_smoothness 0 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 0 \
      --iters 5000

python pipeline/sceneflow.py \
      --exp_name visibility_smoothness5000 \
      --use_smoothness 1 \
      --use_visibility_smoothness 0 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 0 \
      --iters 5000

python pipeline/sceneflow.py \
      --exp_name forward_flow5000 \
      --use_smoothness 1 \
      --use_visibility_smoothness 0 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 1 \
      --iters 5000

python pipeline/sceneflow.py \
      --exp_name forward_and_visibility5000 \
      --use_smoothness 1 \
      --use_visibility_smoothness 1 \
      --use_reverse_nn 0 \
      --use_forward_flow_smoothness 1 \
      --iters 5000

python pipeline/sceneflow.py \
      --exp_name allec5000 \
      --use_smoothness 1 \
      --use_visibility_smoothness 1 \
      --use_reverse_nn 1 \
      --use_forward_flow_smoothness 1 \
      --iters 5000

