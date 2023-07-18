from .kittisf import *

test_idx = list(range(len(all_files)))


data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'proj_H' : proj_H,
               'proj_W' : proj_W,
               'all_files' : all_files,
               'train_idx' : train_idx,
               'test_idx' : test_idx,
               }
