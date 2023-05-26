import numpy as np
import torch
import glob

from data.PATHS import DATA_PATH

class Argo1_NSF():
    # implement waymo
    def __init__(self, root_dir=DATA_PATH + '/argoverse/val/'):
        self.root_dir = root_dir
        self.all_files = glob.glob(root_dir + '/*/*.npz')
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):

        data = np.load(self.all_files[self.idx])

        pc1 = torch.from_numpy(data['pc1']).unsqueeze(0)
        pc2 = torch.from_numpy(data['pc2']).unsqueeze(0)

        gt_flow = torch.from_numpy(data['flow']).unsqueeze(0)

        self.idx += 1

        if self.idx < len(self.all_files):
            return pc1, pc2, gt_flow

        raise StopIteration

    def __len__(self):
        return len(self.all_files)
