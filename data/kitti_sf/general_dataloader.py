import numpy as np
import os
import torch
from data.PATHS import KITTI_SF_PATH

def load_frame(frame_id):

    data_path = f"{KITTI_SF_PATH}/all_data_format/{frame_id:06d}.npz"
    data = np.load(data_path, allow_pickle=True)

    return data

class KittiSF_Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data_all = f"{KITTI_SF_PATH}/all_data_format/"


    def __getitem__(self, idx):

        data = load_frame(idx)

        return data

    def __len__(self):
        return len(os.listdir(f"{KITTI_SF_PATH}/all_data_format/"))
