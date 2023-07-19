# KITTI-SF
import os
import glob
import numpy as np
import torch
# from vis.deprecated_vis import *

# not in lidar frame here!
# ---> to shift to right coordinate system for pitch and yaw

data_path = f'{os.path.expanduser("~")}/data/kittisf/'

lidar_pose = (0,0,0)

all_files = sorted(glob.glob(data_path + 'all_data_format/*.npz'))

len_dataset = len(all_files)

train_idx = [0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33,
                     34, 35, 36, 39, 40, 42, 43, 44, 45, 47, 49, 51, 53, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70,
                     73, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 89, 93, 94, 95, 96, 97, 98, 101, 105, 108, 109, 110,
                     111, 112, 113, 114, 115, 117, 118, 122, 123, 124, 126, 128, 130, 131, 133, 135, 137, 138, 140]

# test_idx = [i for i in range(len_dataset) if i not in train_idx]
# THIS IS SCOOP SPLIT Kitti_T
test_idx = [3, 7, 11, 19, 25, 26, 34, 37, 42, 43, 46, 51, 53, 55, 57, 59, 62, 63, 64, 66,
            68, 76, 77, 79, 80, 85, 94, 95, 97, 98, 105, 112, 113, 115, 116, 117, 119, 120,
            129, 132, 141, 142, 146, 148, 150, 158, 160, 162, 168, 199]

# kitti_O is all of them
# kittit


# data preprocessing
def frame_preprocess(pc1, pc2, gt_flow):
    y_min = -1.4
    z_max = 35
    pc_scene = pc2.copy()
    # DATASET IS BIJECTIVE ...

    above_ground = np.logical_and(pc1[:, 1] > y_min, pc2[:, 1] > y_min)
    is_close = np.logical_and(pc1[:, 2] < z_max, pc2[:, 2] < z_max)

    gt_flow = gt_flow[above_ground & is_close]
    pc1 = pc1[above_ground & is_close]
    pc2 = pc2[above_ground & is_close]

    # transform to lidar coordinates (SCOOP inference does inside already)
    pc1 = pc1[:, [2, 0, 1]]
    pc2 = pc2[:, [2, 0, 1]]
    pc_scene = pc_scene[:, [2, 0, 1]]
    gt_flow = gt_flow[:, [2, 0, 1]]

    # visualize_flow3d(pc1, pc2, gt_flow)

    return pc1, pc2, gt_flow, pc_scene

# P = ['7.188560e+02 0.000000e+00 6.071928e+02 -3.372877e+02 0.000000e+00 7.188560e+02 1.852157e+02 2.369057e+00 0.000000e+00 0.000000e+00 1.000000e+00 4.915215e-03']

# P = ['7.188560e+02 0.000000e+00 6.071928e+02 4.538225e+01 0.000000e+00 7.188560e+02 1.852157e+02 -1.130887e-01 0.000000e+00 0.000000e+00 1.000000e+00 3.779761e-03']
P = ['7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03']
P_rect = np.array([float(i) for i in P[0].split(' ')]).reshape(3,4)
P_rect = torch.from_numpy(P_rect)


# camera params not yet set!!!
fov_up= 20  #
fov_down = -20
H = 376
W = 1242  # maybe multiple by 4 for full fied of view?

data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'H' : H,
               'W' : W,
               'all_files' : all_files,
               'train_idx' : train_idx,
               'test_idx' : test_idx,
               'P_rect' : P_rect,
               }
