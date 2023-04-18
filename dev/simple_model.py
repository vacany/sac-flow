import os.path

import numpy as np
import glob
import torch

from sklearn.neighbors import NearestNeighbors

from vis.deprecated_vis import visualize_points3D, visualize_multiple_pcls, visualize_flow3d
from ops.linalg import synchronize_poses, transform_pc
from ops.visibility import smooth_loss
# from pytorch3d.ops.knn import knn_points

def scipy_NN(x,y, K=1):
    return NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(x).kneighbors(y, return_distance=False)

class RefineFlow(torch.nn.Module):

    def __init__(self, pc1):
        super(RefineFlow, self).__init__()
        self.flow = torch.nn.Parameter(torch.zeros(pc1.shape[0], 3)).requires_grad_(True)

    def forward(self, pc1=None, pc2=None):
        return self.flow

DATA_DIR = os.path.expanduser("~") + "/rci/data/waymo_test/"
# pcs = [np.load(f) for f in sorted(glob.glob(DATA_DIR + "/patchwork/000000/*.npy"))]
pcs = [np.load(f) for f in sorted(glob.glob(DATA_DIR + "/pathwork_nonground/*.npy"))[:2]]
poses = [np.load(f) for f in sorted(glob.glob(DATA_DIR + "/poses/*.npy"))[:2]]



pc1 = pcs[0]
pose1 = poses[0]

g_pcs = [transform_pc(pcs[i], poses[i]) for i in range(len(pcs))]
g_pc1 = transform_pc(pcs[0], poses[0])
g_pc2 = transform_pc(pcs[1], poses[1])


back_pts1 = (np.linalg.inv(pose1) @ g_pc1.T).T
back_pts2 = (np.linalg.inv(pose1) @ g_pc2.T).T

visualize_multiple_pcls(*[back_pts1, back_pts2])

_x = torch.from_numpy(back_pts1[:,:3]).float().requires_grad_(True)
_y = torch.from_numpy(back_pts2[:,:3]).float().requires_grad_(True)
# closer
radius = 20
x = _x[_x.norm(1, dim=1) < radius]
y = _y[_y.norm(1, dim=1) < radius]

visualize_multiple_pcls(*[x, y])



# subsample to have same number of pts
if x.shape[0] > y.shape[0]:
    x = x[torch.randperm(x.shape[0])[:y.shape[0]]]
else:
    y = y[torch.randperm(y.shape[0])[:x.shape[0]]]

model = RefineFlow(x)

x_f = x + model.flow

from data.gpu_utils import get_device
device = get_device()
model = model.to(device)
x = x.to(device)
y = y.to(device)



# knn_points(x_f.unsqueeze(0), y.unsqueeze(0), K=1)


# chamf_dist = (x - y[torch.from_numpy(ind[:,0])]).norm()
# chamf_dist.backward()

print(model.flow.grad)
