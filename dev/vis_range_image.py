import matplotlib.pyplot as plt

from vis.deprecated_vis import *

import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

pc1 = np.load('toy_samples/04_000010.npy')


from pytorch3d.ops.knn import knn_points
from ops.visibility2D import KNN_visibility_solver, substitute_NN_by_mask

# new_nn_ind = substitute_NN_by_mask(nn_ind, valid_KNN)

class VisibilityDepth():

    def __init__(self, pc : np.array, dataset='semantic_kitti', K=8, HOF=2048, VOF=64):

        self.K = K
        self.pc = pc
        self.HOF, self.VOF = HOF, VOF

        # todo apply to argo dataset
        # IDK why billboard is stripped

        # todo add KNN max margin to the solver
        # todo solver 360 degree
        # todo smooth depth in range image as option

        # Metrics for evaluate flow script

        self.update(pc)
    def pc_to_image(self, pc):
        '''
        Args: pc [N, 3]
        Returns: H_coords, V_coords, depth, image_index
        H,V coords are the coordinates of the points in the range image
        depth is the depth of the points in the range image
        image_index is an image with index values in pixel
        '''
        calc_depth = np.linalg.norm(pc[:,:3], axis=1)
        yaw = -np.arctan2(pc[:, 1], pc[:, 0])
        pitch = np.arcsin(pc[:, 2] / (calc_depth + 1e-8))

        # print(pitch)
        # todo this might not be correct, since it is degrees?
        delta_yaw = (yaw.max() - yaw.min()) / self.HOF
        delta_pitch = (pitch.max() - pitch.min()) / self.VOF

        H_coords = np.floor((yaw - yaw.min()) / delta_yaw).astype(int)
        V_coords = np.floor((pitch - pitch.min()) / delta_pitch).astype(int)

        np.clip(H_coords, 0, self.HOF - 1, out=H_coords)
        np.clip(V_coords, 0, self.VOF - 1, out=V_coords)

        depth = - np.ones((self.VOF, self.HOF))
        index_image = - np.ones((self.VOF, self.HOF))

        # visualize_points3D(pc, calc_depth)
        # plt.plot(H_coords, V_coords, '*')
        # plt.axis('equal')
        # plt.savefig('test.png')

        depth[V_coords, H_coords] = calc_depth
        index_image[V_coords, H_coords] = np.arange(calc_depth.shape[0])

        return H_coords, V_coords, depth, index_image

    def update(self, pc, K=None):
        '''
        Will update depth from sorted point cloud - prioritize closest points
        '''
        if K is not None:
            self.K = K

        self.pc = pc[:,:3]

        # inverse indexing to prioritize closest points
        depth_pc = np.linalg.norm(pc[:,:3], axis=1)

        indices = depth_pc.argsort()[::-1]
        sorted_pc1 = pc[indices]

        H_coords, V_coords, depth, index_image = self.pc_to_image(sorted_pc1)

        # visualize_points3D(sorted_pc1, depth_pc[indices])

        self.depth = depth

        plt.imshow(np.flip(depth, axis=0))
        # plt.show()
        plt.savefig('test.png')
        # # GT flow should be sorted backwards

    def KNN_coords(self, pc2, nn_ind):
        '''
        :param pc2: point cloud to strip as Nx3
        :param nn_ind: KNN indices as NxK
        :return: Coordinates of KNN in depth image as NxKx2
        '''
        H_coords, V_coords, _, _ = self.pc_to_image(pc2)

        # plt.imshow(self.depth)
        # plt.savefig('toy_samples/depth.png')

        image_indices = np.stack((V_coords, H_coords)).T
        image_indices = torch.from_numpy(image_indices).to(device)

        KNN_image_indices = image_indices[nn_ind]

        return KNN_image_indices

    def strip_KNN_with_vis(self, pc2, nn_ind):
        '''
        :param pc2: point cloud to strip as Nx3
        :param nn_ind: KNN indices as NxK
        :return: visibility aware KNN indices as NxK (stripped KNNs)
        '''


        KNN_image_indices = self.KNN_coords(pc2, nn_ind)

        depth = torch.from_numpy(self.depth).to(device)
        valid_KNN = KNN_visibility_solver(KNN_image_indices, depth, margin=3)

        visibility_aware_KNN = substitute_NN_by_mask(nn_ind, valid_KNN)

        visualize_one_KNN_in_depth(KNN_image_indices, depth, 8107, K=self.K, margin=10, output_path='test.png')

        return visibility_aware_KNN

    def run_on_self_pc(self):

        torch_pc = torch.from_numpy(self.pc).unsqueeze(0).to(device)
        out_nn = knn_points(torch_pc, torch_pc, K=self.K)
        nn_dist, nn_ind = out_nn[0][0], out_nn[1][0]

        # visualize_points3D(self.pc, np.arange(self.pc.shape[0]))
        visibility_aware_KNN = self.strip_KNN_with_vis(self.pc, nn_ind)

        return visibility_aware_KNN


from data.PATHS import DATA_PATH
argo_data = np.load(DATA_PATH + '/argoverse/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/315968385323560000_315968385423756000.npz')
argo_pc1 = argo_data['pc1']
# todo try it on argo data



pitch = np.arcsin(argo_pc1[:, 2] / (np.linalg.norm(argo_pc1[:,:3], axis=1) + 1e-8))
#          sensor_name        qw        qx        qy        qz      tx_m      ty_m      tz_m
# 9             up_lidar  0.999996  0.000000  0.000000 -0.002848  1.350180  0.000000  1.640420
# 10          down_lidar -0.000089 -0.994497  0.104767  0.000243  1.355162  0.000133  1.565252

from plyfile import PlyData, PlyElement
plydata = PlyData.read('toy_samples/argo_sample.ply')
argo_xyz = plydata.elements[0].data
argo = np.stack((argo_xyz['x'], argo_xyz['y'], argo_xyz['z']), axis=1)
rad_argo = np.linalg.norm(argo[:,:3], axis=1) < 35

visualize_points3D(argo_pc1, pitch)
visualize_points3D(argo[rad_argo], argo[rad_argo,2] > 0.1)

rad_pc1 = np.linalg.norm(pc1[:,:3], axis=1) < 35
visualize_points3D(pc1[rad_pc1], pc1[rad_pc1,2] > -1.4)


# Vis_Depth = VisibilityDepth(pc1[:,:3], K=16)
#
# # H, V, depth_img, _ = Vis_Depth.pc_to_image(pc1[:,:3])
# # depth_img = Vis_Depth.depth
#
# # plt.imshow(np.flip(depth_img, axis=0))
# # plt.axis('equal')
# # plt.savefig('toy_samples/depth.png')
#
# # visualize_points3D(pc1, np.arange(pc1.shape[0]))
# vis_aware_KNN = Vis_Depth.run_on_self_pc()
#
# orig_nn = knn_points(torch.from_numpy(pc1).unsqueeze(0).to(device), torch.from_numpy(pc1).unsqueeze(0).to(device), K=Vis_Depth.K)[1][0]
# # out_nn = knn_points(pc2, pc2, K=self.K)
# # nn_dist, nn_ind = out_nn[0][0], out_nn[1][0]
# # Vis_Depth.strip_KNN_with_vis(pc1, nn_ind)
#
# visualize_KNN_connections(pc1[:,:3], vis_aware_KNN.detach().cpu().numpy())
# visualize_KNN_connections(pc1[:,:3], orig_nn.detach().cpu().numpy())

