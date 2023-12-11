#! /usr/bin/env python

import os
import torch
import numpy as np
from data.path_utils import get_traj, get_inst_trajes
from ops.transform import xyz_axis_angle_to_matrix, matrix_to_xyz_axis_angle

__all__ = [
    'path_smoothness',
    'path_smoothness_batched',
]


def path_smoothness(poses, dt=1.):
    assert isinstance(poses, torch.Tensor)
    assert poses.ndim == 2 or (poses.ndim == 3 and poses.shape[1:] == (4, 4))  # (N, 6) or (N, 4, 4)
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        xyza = matrix_to_xyz_axis_angle(poses)
    else:
        xyza = poses

    n_poses = len(xyza)
    # assert n_poses >= 3, 'Path must contain at least 3 poses to compute acceleration'
    if n_poses < 3:
        print('Path must contain at least 3 poses to estimate acceleration for smoothness loss')
        cost = torch.zeros_like(xyza).sum()
        return cost
    # xyza is a tensor of shape (N, 6)
    # where N is the number of poses
    # and each pose is a 6D vector (x, y, z, ax, ay, az)
    assert isinstance(xyza, torch.Tensor)
    assert xyza.shape[-1] == 6

    # estimate linear and angular velocities
    d_xyza = torch.diff(xyza, dim=0) / dt
    # l1_y = torch.sum(torch.square(d_xyza[:, 1]))
    # l1_rot = torch.sum(torch.square(d_xyza[:, 3:]))

    # estimate linear and angular accelerations
    dd_xyza = torch.diff(d_xyza, dim=0) / dt
    l2 = torch.sum(torch.square(dd_xyza[:, :3]))
    l2_rot = torch.sum(torch.square(dd_xyza[:, 3:]))

    # 3-rd order smoothness
    # if n_poses > 3:
    #     ddd_xyza = torch.diff(dd_xyza, dim=0) / dt
    #     l3 = torch.sum(torch.square(ddd_xyza[:, :3]))

    # # estimate headings differences from robot's X-axis
    # head_dirs = xyza[1:, :3] - xyza[:-1, :3]
    # head_dirs = head_dirs / torch.norm(head_dirs, dim=1, keepdim=True)
    # # robot's X-axis in world frame
    # robot_x = torch.tensor([1., 0., 0.], dtype=xyza.dtype, device=xyza.device)
    # # transform robot's X-axis to world frame
    # R = xyz_axis_angle_to_matrix(xyza[:-1])[:, :3, :3]
    # robot_x = R @ robot_x
    # # rotation difference between robot's X-axis and heading direction
    # l_head = (1 - torch.square(torch.sum(robot_x * head_dirs, dim=1)).mean()).abs()

    # estimate cost
    # cost = l2 + l3 + l1_rot + l2_rot + l_head
    # cost = l2 + l3 + l_head
    # cost = l_head
    # cost = l2 + l3 + l2_rot
    cost = l2 + l2_rot
    assert cost >= 0

    return cost


def path_smoothness_batched(poses, dt=1., reduce=True):
    assert len(poses) > 0
    assert isinstance(poses[0], torch.Tensor)
    assert (poses[0].ndim == 2 and poses[0].shape[1] == 6) or (poses[0].ndim == 3 and poses[0].shape[1:] == (4, 4))  # (N, 6) or (N, 4, 4)

    B = len(poses)
    costs = []
    for i in range(B):
        cost = path_smoothness(poses[i], dt=dt)
        costs.append(cost)
    costs = torch.stack(costs, dim=0)
    if reduce:
        return costs.mean()
    return costs


def demo():
    from mayavi import mlab
    from vis.mayavi_interactive import draw_coord_frames
    from data.dataloader import SFDataset4D
    from data.PATHS import DATA_PATH

    np.random.seed(0)
    torch.manual_seed(0)

    # get trajectories from dataset
    ds = SFDataset4D(dataset_type='waymo', n_frames=10)
    # ds_i = np.random.randint(0, len(ds))
    ds_i = 0
    data_sample = ds[ds_i]

    xyza = get_traj(data_sample, noise=0.2, axis_angle=True)
    xyza.requires_grad_(True)

    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    xyza_init = xyza.detach().clone()
    # optimize trajectory to have it smooth
    optimizer = torch.optim.Adam([xyza], lr=0.01)
    n_iters = 100

    for i in range(n_iters):
        optimizer.zero_grad()
        loss = path_smoothness(xyza)
        loss.backward()
        optimizer.step()
        print('iter {}, loss: {:.4f}'.format(i, loss.item()))

        if i % 5 == 0 or i == n_iters - 1:
            with torch.no_grad():
                # visualize traj in mayavi
                mlab.clf()
                mlab.title('iter {}, loss: {:.4f}'.format(i, loss.item()), size=0.5, color=(0, 0, 0))
                mlab.points3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 1, 0), scale_factor=0.1)
                mlab.plot3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 1, 0), tube_radius=0.01)
                draw_coord_frames(xyz_axis_angle_to_matrix(xyza), scale=0.5)

                # draw initial traj
                mlab.points3d(xyza_init[:, 0], xyza_init[:, 1], xyza_init[:, 2], color=(0, 0, 1), scale_factor=0.1)
                mlab.plot3d(xyza_init[:, 0], xyza_init[:, 1], xyza_init[:, 2], color=(0, 0, 1), tube_radius=0.01)
                fig.scene._lift()
    mlab.show()


def demo_multipath(vis=True):
    from mayavi import mlab
    from vis.mayavi_interactive import draw_coord_frames
    from data.dataloader import SFDataset4D
    from data.PATHS import DATA_PATH

    ds = SFDataset4D(root_dir=os.path.join(DATA_PATH, 'sceneflow'), dataset_type='waymo', n_frames=20)

    # n_trajes = 20
    # poses = [get_traj(data_sample=ds[ds_i], noise=0.2) +
    #          torch.tensor([0., 0., k * 1., 0., 0., 0.]) for k, ds_i in enumerate(np.random.choice(len(ds), n_trajes, replace=False))]

    # get instance trajectories from dataset
    ds_i = np.random.randint(0, len(ds))
    data_sample = ds[ds_i]
    _, poses = get_inst_trajes(data_sample, min_traj_len=0.5, noise=0.2, axis_angle=True, verbose=True)

    # make poses differentiable
    poses = [p.requires_grad_(True) for p in poses]

    # optimize trajectories to have them smooth
    lr = 0.01
    optimizer = torch.optim.Adam(poses, lr=lr)

    if vis:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    n_iters = 100
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = path_smoothness_batched(poses)
        loss.backward()
        optimizer.step()
        print('iter {}, loss: {:.4f}'.format(i, loss.item()))

        if vis and (i % 5 == 0 or i == n_iters - 1):
            with torch.no_grad():
                mlab.clf()
                # draw initial traj
                mlab.title('Initial trajectories', size=0.5, color=(0, 0, 0))
                for xyza in poses:
                    mlab.points3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 0, 1), scale_factor=0.1)
                    mlab.plot3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 0, 1), tube_radius=0.05)
                    # draw_coord_frames(xyz_axis_angle_to_matrix(xyza), scale=0.5)
                # set camera
                # mlab.view(azimuth=0, elevation=0, distance=100, focalpoint=(0, 0, 0))
                fig.scene._lift()
    mlab.show()


def main():
    demo()
    # demo_multipath(vis=True)


if __name__ == '__main__':
    main()
