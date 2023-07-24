
import glob
import os
import numpy as np
from data.PATHS import DATA_PATH

data_path = f'{DATA_PATH}/sceneflow/waymo/'
all_files = sorted(glob.glob(data_path + 'val/*/*.npz'))
len_dataset = len(all_files)


# train_idx = []
test_idx = [i for i in range(len_dataset)]


LIDAR_LOCAL_POSE = np.array([[-8.54212716e-01, -5.19923095e-01, -7.81823797e-04, 1.43000000e+00],
                             [ 5.19918423e-01, -8.54209872e-01,  3.21373964e-03,  0.00000000e+00],
                             [-2.33873907e-03,  2.33873267e-03,  9.99994530e-01,  2.18400000e+00],  # this height is also upper plane of ego box
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

lidar_pose = (1.43, 0, 2.184)


fov_up= 2.4
fov_down = -17.6
H = 64 #?
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


    return pc1, pc2, gt_flow, pc2.copy()
