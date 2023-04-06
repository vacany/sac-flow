import os.path

import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import sys
import socket

from scipy import interpolate




if socket.gethostname().startswith("Pat"):
    sys.path.append('/home/patrik/.local/lib/python3.8/site-packages')
    import pptk

    def visualize_points3D(points, labels=None, point_size=0.01):
        if not socket.gethostname().startswith("Pat"):
            return

        if type(points) is not np.ndarray:
            points = points.detach().cpu().numpy()

        if type(labels) is not np.ndarray and labels is not None:
            labels = labels.detach().cpu().numpy()

        if labels is None:
            v = pptk.viewer(points[:,:3])
        else:
            v = pptk.viewer(points[:, :3], labels)
        v.set(point_size=point_size)

        return v

    def visualize_pcd(file, point_size=0.01):
        import open3d
        points = np.asarray(open3d.io.read_point_cloud(file).points)
        v=pptk.viewer(points[:,:3])
        v.set(point_size=point_size)

    def visualize_voxel(voxel, cell_size=(0.2, 0.2, 0.2)):
        x,y,z = np.nonzero(voxel)
        label = voxel[x,y,z]
        pcl = np.stack([x / cell_size[0], y / cell_size[1], z / cell_size[2]]).T
        visualize_points3D(pcl, label)

    def visualize_poses(poses):
        xyz = poses[:,:3,-1]
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(xyz[:,0], xyz[:,1])
        res = np.abs(poses[:-1, :3, -1] - poses[1:, :3, -1]).sum(1)
        axes[1].plot(res)
        plt.show()

    def visualize_multiple_pcls(*args, **kwargs):
        p = []
        l = []

        for n, points in enumerate(args):
            if type(points) == torch.Tensor:
                p.append(points[:,:3].detach().cpu().numpy())
            else:
                p.append(points[:,:3])
            l.append(n * np.ones((points.shape[0])))

        p = np.concatenate(p)
        l = np.concatenate(l)
        v=visualize_points3D(p, l)
        v.set(**kwargs)

    def visualize_plane_with_points(points, n_vector, d):

        xx, yy = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), 100),
                             np.linspace(points[:,1].min(), points[:,1].max(), 100))

        z = (- n_vector[0] * xx - n_vector[1] * yy - d) * 1. / n_vector[2]
        x = np.concatenate(xx)
        y = np.concatenate(yy)
        z = np.concatenate(z)

        plane_pts = np.stack((x, y, z, np.zeros(z.shape[0]))).T

        d_dash = - n_vector.T @ points[:,:3].T

        bin_points = np.concatenate((points, (d - d_dash)[:, None]), axis=1)

        vis_pts = np.concatenate((bin_points, plane_pts))

        visualize_points3D(vis_pts, vis_pts[:,3])


    def visualize_flow3d(pts1, pts2, frame_flow):
        # flow from multiple pcl vis
        # valid_flow = frame_flow[:, 3] == 1
        # vis_flow = frame_flow[valid_flow]
        # threshold for dynamic is flow larger than 0.05 m
        dist_mask = np.sqrt((frame_flow[:,:3] ** 2).sum(1)) > 0.05

        vis_pts = pts1[dist_mask,:3]
        vis_flow = frame_flow[dist_mask]

        # todo color for flow estimate
        # for raycast
        # vis_pts = pts1[valid_flow, :3]
        # vis_pts = pts1[dist_mask, :3]

        all_rays = []
        # breakpoint()
        for x in range(int(20)):
            ray_points = vis_pts + (vis_flow[:, :3]) * (x / int(20))
            all_rays.append(ray_points)

        all_rays = np.concatenate(all_rays)

        visualize_multiple_pcls(*[pts1, all_rays, pts2], point_size=0.02)

    def visualizer_transform(p_i, p_j, trans_mat):
        '''

        :param p_i: source
        :param p_j: target
        :param trans_mat: p_i ---> p_j transform matrix
        :return:
        '''

        p_i = np.insert(p_i, obj=3, values=1, axis=1)
        vis_p_i = p_i @ trans_mat.T
        visualize_multiple_pcls(*[p_i, vis_p_i, p_j])

else:
    def visualize_points3D(pts, color=None, path=os.path.expanduser("~") + '/data/tmp_vis/visul'):
        np.save(path, pts)

        if color is None:
            np.save(path + '_color.npy', np.zeros(pts.shape[0], dtype=bool))
        else:
            np.save(path + '_color.npy', color)


        pass

# matplotlib
def visualize_connected_points(pts1, pts2, title=None, savefig=None):
    plt.plot(pts1[:, 0], pts1[:, 1], 'ob')
    plt.plot(pts2[:, 0], pts2[:, 1], 'or')

    for i in range(len(pts1)):
        p = pts1[i]
        r = pts2[i]

        diff = r - p
        yaw_from_meds = np.arctan2(diff[1], diff[0])
        yaw_degree = 180 * yaw_from_meds / np.pi

        plt.plot(p[0], p[1], '.b')
        plt.plot(r[0], r[1], '.r')

        plt.annotate(f"{yaw_degree:.2f} deg", p[:2] + (0.01, 0))

        connection = np.array([(p[0], p[1]), (r[0], r[1])])
        plt.plot(connection[:, 0], connection[:, 1], 'g--')

    if title is not None:
        plt.title(title)
    plt.axis('equal')
    plt.show()

# def visualize_flow3d(pts, velocity, savefig=None):
#
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     for idx in range(len(pts)):
#         ax.quiver(pts[idx, 0], pts[idx, 1], pts[idx, 2],  # <-- starting point of vector
#                   velocity[idx, 0], velocity[idx, 1], velocity[idx, 2],  # <-- directions of vector
#                   color='red', alpha=.6, lw=2,
#                   )
#     plt.show()
#
# def plot_points3d(pts, features, lookat=None, title="Point Cloud Lookat", save=None):
#
#     fig = plt.figure(figsize=(5, 5), dpi=200)
#     ax = fig.add_subplot(projection='3d')
#
#     max_r = 20
#     filter_pts = pts.copy()
#
#     # mask =
#     # mask = min_square_by_pcl(filter_pts, lookat[None,:], extend_dist=(max_r, max_r, max_r), return_mask=True)
#
#     xs = filter_pts[mask, 0]
#     ys = filter_pts[mask, 1]
#     zs = filter_pts[mask, 2]
#
#     ax.scatter(xs, ys, zs, marker='.', s=2, c=features[mask], alpha=0.8, cmap='jet', vmin=0, vmax=1)
#     ax.set_xlim([-.5 + lookat[0], 6 + lookat[0]])
#     ax.set_ylim([-4 + lookat[1], 4 + lookat[1]])
#     ax.set_zlim([-4 + lookat[2], 4 + lookat[2]])
#
#     ax.view_init(elev=25, azim=210)
#     ax.dist = 6
#     colormap = plt.cm.get_cmap('jet', 10)
#
#     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), orientation='vertical')
#     plt.title(title)
#     plt.axis('off')
#
#     if save is not None:
#         plt.savefig(save)
#     else:
#         plt.show()
#
#     plt.close()

def fit_3D_spline():
    # 3D example
    total_rad = 10
    z_factor = 3
    noise = 0.1

    num_true_pts = 200
    s_true = np.linspace(0, total_rad, num_true_pts)
    x_true = np.cos(s_true)
    y_true = np.sin(s_true)
    z_true = s_true/z_factor

    num_sample_pts = 80
    s_sample = np.linspace(0, total_rad, num_sample_pts)
    x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)

    tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(x_true, y_true, z_true, 'b')
    ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    fig2.show()
    plt.show()
