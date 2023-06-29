#!/bin/bash

# shellcheck disable=SC2164
cd $HOME/pcflow

GPU=4
# probably split to paralelize

python pipeline/sceneflow.py \
      --dataset argoverse \
      --exp_name ff \
      --lr 0.008 0.01 \
      --gpu $GPU \
      \
      --l_ch 1 \
      --l_dt 0 \
      --l_sm 1 \
      --l_ff 1 \
      --l_vsm 0 \
      \
      --l_ch_bothways 0 \
      --smooth_K 4 8 12 \
      --KNN_max_radius 1 1.5 \
      --pc2_smoothness 0 1




# Chamfer
#parser.add_argument('--loss_chamfer_weight', type=float, default=0.) #?
#parser.add_argument('--loss_chamfer_both_ways', type=bool, default=0) #?
#parser.add_argument('--loss_chamfer_use_normals', type=bool, default=0) #?
#parser.add_argument('--loss_chamfer_normals_K', type=int, default=4) #?
## parser.add_argument('--loss_chamfer_use_visibility', type=bool, default=1) # Freespace?
## DT
#parser.add_argument('--loss_DT_weight', type=float, default=1) #?
#parser.add_argument('--loss_DT_grid_factor', type=int, default=10) #?
#
## Smoothness
#parser.add_argument('--loss_smooth_weight', type=float, default=0) #?
#parser.add_argument('--loss_smooth_normals', type=int, default=0) #?
## Visibility-aware smoothness
#parser.add_argument('--loss_K_normals', type=int, default=4) #?
#parser.add_argument('--loss_vis_weight', type=float, default=0) #?
#parser.add_argument('--loss_KNN_max_radius', type=float, default=1.5) #?
#parser.add_argument('--loss_KNN_smooth', type=int, nargs='+', default=8)
#parser.add_argument('--use_visibility_smoothness', type=int, default=0) #?
#
## Forward flow smoothness
#parser.add_argument('--loss_FF_weight', type=float, default=0) #?
#parser.add_argument('--loss_FF_K', type=int, default=0)  # ?
#
#### Hyperparameters
#parser.add_argument('--lr', type=float, default=0.008, help='learning rate') #?
#parser.add_argument('--iters', type=int, nargs='+', default=5000, help='number of iterations')  # ?
#parser.add_argument('--early_patience', type=int, default=10, help='when to consider convergence') #?
#parser.add_argument('--early_min_delta', type=float, default=0.001, help='convergence difference') #?
#
