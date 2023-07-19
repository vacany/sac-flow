# ARGOVERSE
# https://www.argoverse.org/av1.html
# https://argoverse.github.io/user-guide/datasets/sensor.html
# https://arxiv.org/pdf/1911.02620.pdf
import glob
import os
import numpy as np
#                                                       t_x       t_y       t_z
#  up_lidar  0.999996  0.000000  0.000000 -0.002848  1.350180  0.000000  1.640420
#  down_lidar -0.000089 -0.994497  0.104767  0.000243  1.355162  0.000133  1.565252

# not in lidar frame here!
# ---> to shift to right coordinate system for pitch and yaw

data_path = f'{os.path.expanduser("~")}/data/argoverse/'
all_files = sorted(glob.glob(data_path + 'val/*/*.npz'))
len_dataset = len(all_files)


train_idx = []
test_idx = [i for i in range(len_dataset) if i not in train_idx]



t_x1 = 1.350180
t_x2 = 1.355162
t_x_mean = (t_x1 + t_x2) / 2

t_y1 = 0.000000
t_y2 = 0.000133
t_y_mean = (t_y1 + t_y2) / 2

t_z1 = 1.640420
t_z2 = 1.565252
t_z_mean = (t_z1 + t_z2) / 2

lidar_pose = (t_x_mean, t_y_mean, t_z_mean)

# glob.glob(root_dir + '/' + dataset_type + f'/{subfold}/*/*.npz')

fov_up= 25
fov_down = -25
H = 64 #?
W = 2048

data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'H' : H,
               'W' : W,
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


    return pc1, pc2, gt_flow, pc2.copy()
