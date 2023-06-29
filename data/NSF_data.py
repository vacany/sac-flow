import numpy as np
import torch
import glob

from data.PATHS import DATA_PATH
import importlib
class NSF_dataset():

    def __init__(self, root_dir=DATA_PATH, dataset_type : str = 'argoverse', subfold='val'):

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.subfold = subfold
        self.all_files = glob.glob(root_dir + '/' + dataset_type + f'/{subfold}/*/*.npz')
        self.idx = 0

        if dataset_type in ['argoverse']:
            dataset_module = importlib.import_module('data.params.' + dataset_type)
            self.data_config = dataset_module.data_config

    def __iter__(self):
        return self

    def __next__(self):

        if self.idx == len(self.all_files):
            raise StopIteration

        data = np.load(self.all_files[self.idx])

        pc1 = torch.from_numpy(data['pc1']).unsqueeze(0).to(torch.float32)
        pc2 = torch.from_numpy(data['pc2']).unsqueeze(0).to(torch.float32)

        gt_flow = torch.from_numpy(data['flow']).unsqueeze(0).to(torch.float32)

        self.idx += 1

        return pc1, pc2, gt_flow


    def __len__(self):
        return len(self.all_files)

if __name__ == '__main__':
    dataset = NSF_dataset(dataset_type='nuscenes', subfold='val')

    for i in dataset:
        print(i[0].shape, i[1].shape, i[2].shape)
