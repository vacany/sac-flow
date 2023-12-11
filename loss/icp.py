#! /usr/bin/env python

import torch
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree


def point_to_point_dist(clouds, icp_inlier_ratio=0.8, differentiable=True, verbose=False):
    """ICP-like point to point distance.

    Computes point to point distances for consecutive pairs of point cloud scans, and returns the average value.

    :param clouds: List of clouds. Individual scans from a data sequences.
    :param icp_inlier_ratio: Ratio of inlier points between a two pairs of neighboring clouds.
    :param verbose:
    :return:
    """
    assert 0.0 <= icp_inlier_ratio <= 1.0

    point2point_dist = torch.tensor(0.0, dtype=clouds[0].dtype, device=clouds[0].device)
    n_clouds = len(clouds) - 1
    for i in range(n_clouds):
        points1 = clouds[i]
        points2 = clouds[i + 1]

        points1 = torch.as_tensor(points1, dtype=torch.float)
        points2 = torch.as_tensor(points2, dtype=torch.float)
        assert not torch.all(torch.isnan(points1))
        assert not torch.all(torch.isnan(points2))

        # find intersections between neighboring point clouds (1 and 2)
        if not differentiable:
            tree = cKDTree(points2)
            dists, ids = tree.query(points1.detach(), k=1)
        else:
            dists, ids, _ = knn_points(points1[None], points2[None], K=1)
            dists = torch.sqrt(dists).squeeze()
            ids = ids.squeeze()
        dists = torch.as_tensor(dists)
        dist_th = torch.nanquantile(dists, icp_inlier_ratio)
        mask1 = dists <= dist_th
        mask2 = ids[mask1]
        inl_err = dists[mask1].mean()

        points1_inters = points1[mask1]
        assert len(points1_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"
        points2_inters = points2[mask2]
        assert len(points2_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"

        # point to point distance
        vectors = points2_inters - points1_inters
        point2point_dist += torch.linalg.norm(vectors, dim=1).mean()

        if verbose:
            if inl_err > 0.3:
                print('ICP inliers error is too big: %.3f (> 0.3) [m] for pairs (%i, %i)' % (inl_err, i, i + 1))

            print('Mean point to point distance: %.3f [m] for scans: (%i, %i), inliers error: %.6f' %
                  (point2point_dist.item(), i, i+1, inl_err.item()))

    point2point_dist = torch.as_tensor(point2point_dist / n_clouds)
    return point2point_dist


def demo():
    import os
    from data.PATHS import DATA_PATH
    from data.dataloader import SFDataset4D
    import open3d as o3d
    from matplotlib import pyplot as plt
    from ops.filters import filter_range, filter_grid

    N = 60
    ds = SFDataset4D(dataset_type='argoverse', data_split='train4', n_frames=N - 1)
    i = 0
    poses12 = ds[i]['relative_pose']
    clouds = ds[i]['pc1']
    masks = ds[i]['padded_mask_N']

    # construct path from relative poses
    pose = torch.eye(4)[None]
    poses = pose.clone()
    for i in range(len(poses12)):
        pose = pose @ torch.linalg.inv(poses12[i])
        poses = torch.cat([poses, pose], dim=0)

    # transform point clouds to the same frame (of a first point cloud)
    clouds_path = []
    for i in range(len(clouds)):
        pose = poses[i]
        cloud = clouds[i]
        cloud = cloud[masks[i]]
        # filter point cloud
        # cloud = filter_depth(cloud, min=1., max=25.0)
        # cloud = filter_grid(cloud, grid_res=0.3)
        # transform point cloud
        cloud = cloud @ pose[:3, :3].T + pose[:3, 3][None]
        clouds_path.append(cloud)

    # icp loss
    icp_fn = lambda x: point_to_point_dist(x, icp_inlier_ratio=0.9, differentiable=False)
    icp_loss = icp_fn(clouds)
    print('ICP loss: %.3f [m]' % icp_loss.item())
    icp_loss_global = icp_fn(clouds_path)
    print('ICP loss for transformed clouds: %.3f [m]' % icp_loss_global.item())

    # show path and point clouds
    clouds_path = torch.cat(clouds_path, dim=0).numpy()
    plt.figure()
    plt.plot(poses[:, 0, 3], poses[:, 1, 3], 'o-')
    # plot point clouds
    plt.plot(clouds_path[:, 0], clouds_path[:, 1], '.', markersize=1)
    plt.axis('equal')
    plt.grid()
    plt.show()

    # visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clouds_path)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    demo()
