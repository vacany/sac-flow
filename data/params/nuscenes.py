# https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf
import glob
import os
import numpy as np


data_path = f'{os.path.expanduser("~")}/data/nuscenes/'
all_files = sorted(glob.glob(data_path + 'val/*/*.npz'))
len_dataset = len(all_files)


# train_idx = []
test_idx = [i for i in range(len_dataset)]

lidar_pose = (0.943713, 0.0, 1.84023)   # from calibrated sensor

# Spinning, 32 beams, 20Hz capture frequency, 360◦
# horizontal FOV, −30◦ to 10◦ vertical FOV, ≤ 70m
# range, ±2cm accuracy, up to 1.4M points per second

fov_up= 10
fov_down = -30
H = 32 #?
W = 2048

data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'H' : H,
               'W' : W
               }



def frame_preprocess(pc1, pc2, gt_flow):
    # These are already preprocessed
    # y_min = -1.4
    # z_max = 35
    # gt_flow = gt_flow[np.logical_and(pc1[:, 2] < z_max, pc1[:, 1] > y_min)]
    # pc1 = pc1[pc1[:, 1] > y_min]
    # pc2 = pc2[pc2[:, 1] > y_min]
    # pc1 = pc1[pc1[:, 2] < z_max]
    # pc2 = pc2[pc2[:, 2] < z_max]


    return pc1, pc2, gt_flow
