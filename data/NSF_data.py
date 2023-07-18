import numpy as np
import torch
import glob

from data.PATHS import DATA_PATH
import importlib
class NSF_dataset():
# dataset type kittisf
    def __init__(self, root_dir=DATA_PATH, dataset_type : str = 'argoverse', subfold='val'):

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.subfold = subfold

        self.idx = 0
        dataset_module = importlib.import_module('data.params.' + dataset_type)
        self.data_config = dataset_module.data_config
        # pre-process data, shift them if not in origin
        self.preprocess_func = dataset_module.frame_preprocess

        # indices = np.array(self.data_config['test_idx'], dtype=np.int32)
        self.all_files = [dataset_module.all_files[idx] for idx in dataset_module.test_idx]

        self.lidar_pose = torch.tensor(self.data_config['lidar_pose']).unsqueeze(0).to(torch.float32)

    def __iter__(self):
        return self

    def __next__(self):

        if self.idx == len(self.all_files):
            raise StopIteration

        data = np.load(self.all_files[self.idx])

        pc1 = data['pc1']
        pc2 = data['pc2']
        gt_flow = data['flow']

        pc1, pc2, gt_flow = self.preprocess_func(pc1, pc2, gt_flow)

        pc1 = torch.from_numpy(pc1).unsqueeze(0).to(torch.float32)
        pc2 = torch.from_numpy(pc2).unsqueeze(0).to(torch.float32)

        pc1 = pc1 - self.lidar_pose
        pc2 = pc2 - self.lidar_pose

        gt_flow = torch.from_numpy(gt_flow).unsqueeze(0).to(torch.float32)

        self.idx += 1

        return pc1, pc2, gt_flow


    def __len__(self):
        return len(self.all_files)

if __name__ == '__main__':
    dataset = NSF_dataset(dataset_type='nuscenes', subfold='val')

    for i in dataset:
        print(i[0].shape, i[1].shape, i[2].shape)
