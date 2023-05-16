import torch
from pytorch3d.ops.knn import knn_points
from sklearn.datasets import make_blobs
from vis import *

df, y = make_blobs(n_samples=200, centers=7,n_features=3,random_state=999,cluster_std=0.5)
df[:,2] = 0


# synthetize 3d point based on 2D bounding box and number of points
def create_dash_line_3D(center, dim_ratio=np.array([1.2,0.4,0.05]), N=100):

    df = np.random.uniform(-1,1,(N,3)) * dim_ratio + center

    return df

center = np.array([10,5,0.1])
pcl = []
for i in range(10):
    center[0] += 2

    line_points = create_dash_line_3D(center, dim_ratio=np.array([0.6,0.2,0.02]), N=100 - i * 4)
    line_points = np.insert(line_points, 3, i, axis=1)
    pcl.append(line_points)

pcl = np.concatenate(pcl, axis=0)
# visualize_points3D(pcl, pcl[:,3])


x = torch.from_numpy(pcl).unsqueeze(0).float()
y = torch.from_numpy(pcl[:,3]).unsqueeze(0)
K = 12
max_radius = 1  # this is important for dummy case
class InstanceSegRefiner(torch.nn.Module):

    def __init__(self, x, max_instances=30):
        self.pred = torch.rand(x.shape[0], x.shape[1], max_instances, requires_grad=True)
        super().__init__()

    def forward(self):
        return torch.softmax(self.pred, dim=2)

def mask_split(tensor, indices):
    unique = torch.unique(indices)
    return [tensor[0][indices == i] for i in unique]

def svd_tensor_list(tensor_list):

    U_list = []
    S_list = []
    V_list = []
    for tensor in tensor_list:
        tensor_center = tensor - tensor.mean(dim=0)
        U, S, V = torch.svd(tensor_center)
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    return U_list, torch.stack(S_list), torch.stack(V_list)


model = InstanceSegRefiner(x)


# data argo
# argo_pc = np.load(f"{os.path.expanduser('~')}/rci/data/argoverse2/sensor/train/71283e26-905b-3811-b9e0-c10c0253769b/lidar/000069.npy")
# semkit = np.load(f"{os.path.expanduser('~')}/rci/data/semantic_kitti/Rigid3DSceneFlow/semantic_kitti/00/000453_000454.npz")
# waymo_pc = np.load(f"{os.path.expanduser('~')}/rci/data/waymo/training/segment-17677899007099302421_5911_000_5931_000_with_camera_labels.tfrecord/lidar/000030.npy")

# dejvice_pc = np.load(f"{os.path.expanduser('~')}/Downloads/dejvice_data/data/pcl/000659.npz")['pcl']

# visualize_points3D(argo_pc, argo_pc[:,3])
# visualize_points3D(dejvice_pc, dejvice_pc[:,3])
# visualize_points3D(semkit['pc1'], semkit['sem_label_s'])
# visualize_points3D(waymo_pc, waymo_pc[:,3] > 0.5)

# klane

import numpy as np

from plyfile import PlyData, PlyElement
plydata = PlyData.read('/home/patrik/Downloads/toronto3d/Toronto_3D/L001.ply')

all_times = plydata.elements[0].data['scalar_GPSTime']
first_time = plydata.elements[0].data['scalar_GPSTime'].min()

plydata.elements[0].data['x'][all_times == 324786.78]




dist, nn_ind, _ = knn_points(x, x, K=K)
tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(nn_ind.device)
nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]

mask = model.pred
min_nbr_pts = 5
loss_norm = 1
optimizer = torch.optim.Adam([mask], lr=0.1)

# ~ diffentiable DBSCAN
# min distance from clusters

for e in range(1000):
    # still not batch-wise
    out = mask[0][nn_ind[0]]
    out = out.permute(0, 2, 1)
    out = out.unsqueeze(0)

    # norm for each of N separately
    smooth_loss = (mask.unsqueeze(3) - out).norm(p=loss_norm, dim=2).mean()

    u, c = torch.unique(mask.argmax(dim=2), return_counts=True)


    small_clusters = u[c < min_nbr_pts]
    small_cluster_loss = (mask[0, :, small_clusters] ** 2).mean()

    # add differences inside one cluster to disconnect them?
    pseudo_max = mask.max(dim=2)[1]

    # Structure loss? - compact clusters
    dist_from_mean = (x[0,:, :3][pseudo_max[0]==10] - x[0,:,:3][pseudo_max[0]==10].mean(0)).norm(dim=1)


    # increase likelihood as pseudo-label? not well designed, but does something, is good when smaller weights

    pseudo_xe_loss = torch.nn.functional.cross_entropy(mask.permute(0,2,1), pseudo_max)

    loss = smooth_loss + small_cluster_loss.mean() + 0.1 * pseudo_xe_loss #

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"{e} ---> {smooth_loss.item()}")

predicted_mask = mask.argmax(dim=2).detach().cpu().numpy()

from mayavi import mlab

figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
# nodes = mlab.points3d(x[0, :, 0], x[0, :, 1], x[0, :, 2], y[0], scale_factor=0.5, scale_mode='none', colormap='jet')
nodes = mlab.points3d(x[0, :, 0], x[0, :, 1], x[0, :, 2], predicted_mask[0], scale_factor=0.1, scale_mode='none', colormap='jet')


mlab.show()



figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))

for idx in torch.unique(y):
    A = x[0][y[0] == idx]
    A_centered = A - A.mean(dim=0)
    U, S, V = torch.svd(A_centered)



    vis_pc = A.detach().cpu().numpy()
    vis_eigen_vectors = V.detach().cpu().numpy().copy()
    vis_eigen_vectors[0, :] = S[0] * vis_eigen_vectors[0, :]
    vis_eigen_vectors[1, :] = S[1] * vis_eigen_vectors[1, :]
    vis_eigen_vectors[2, :] = S[2] * vis_eigen_vectors[2, :]


    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    mlab.points3d(A[:,0], A[:,1], A[:,2], color=(0,0,1), scale_factor=0.1)
    mlab.quiver3d(A[:,0].mean().repeat(3), A[:,1].mean().repeat(3), A[:,2].mean().repeat(3), vis_eigen_vectors[:,0], vis_eigen_vectors[:,1], vis_eigen_vectors[:,2], color=(0,1,0), scale_factor=1)

mlab.show()
