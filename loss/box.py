import torch
from vis.mayavi_interactive import draw_coord_frames, draw_cloud
from mayavi import mlab
from ops.transform import xyz_rpy_to_matrix, matrix_to_xyz_rpy, xyz_yaw_to_matrix
from ops.filters import filter_box, filter_grid
from loss.path import path_smoothness_batched
from pytorch3d.ops import box3d_overlap
from data.box_utils import Box
from data.path_utils import get_inst_trajes, trajlen
from data.box_utils import get_inst_bbox_sizes

__all__ = [
    'bboxes_coverage',
    'bboxes_iou',
    'bboxes_optimization',
]

def sigmoid(x, k=1.):
    # x - input
    # k - steepness
    return 1. / (1. + torch.exp(-k * x))


def bboxes_coverage(points, bbox_poses, bbox_lhws, weights=None, sigmoid_slope=1., return_mask=False, reduce_rewards=True):
    # points: (n_pts, 3)
    # bbox_lhws: (..., 3)
    # bbox_poses: (..., 4, 4)
    assert points.ndim == 2
    assert points.shape[-1] == 3
    assert bbox_lhws.shape[-1] == 3
    assert bbox_poses.shape[-2:] == (4, 4)

    n_pts = len(points)
    bbox_poses = bbox_poses.view((-1, 4, 4))
    bbox_lhws = bbox_lhws.view((-1, 1, 3))
    assert weights is None or weights.shape == (n_pts,)

    # transform points to bbox frame
    points_bbox_frame = (points[None] - bbox_poses[..., :3, 3:4].transpose(-2, -1)) @ bbox_poses[..., :3, :3]
    assert points_bbox_frame.shape[-2:] == (n_pts, 3)
    # assert not torch.any(torch.isnan(points_bbox_frame))

    # https://openaccess.thecvf.com/content/WACV2023/papers/Deng_RSF_Optimizing_Rigid_Scene_Flow_From_3D_Point_Clouds_Without_WACV_2023_paper.pdf
    s1 = sigmoid(-points_bbox_frame - bbox_lhws / 2, k=sigmoid_slope)
    s2 = sigmoid(-points_bbox_frame + bbox_lhws / 2, k=sigmoid_slope)
    rewards = (s2 - s1).prod(dim=-1)
    rewards = torch.clamp(rewards, 0., 1.)
    assert rewards.shape[-1:] == (n_pts,)

    # merge rewards and masks
    rewards = merge_rewards(rewards)
    if weights is not None:
        rewards = rewards * weights
    # assert torch.all(rewards >= 0) and torch.all(rewards <= 1)
    if reduce_rewards:
        rewards = rewards.mean()

    if return_mask:
        # mask of points inside bounding boxes
        mask = ((points_bbox_frame > -bbox_lhws / 2) & (points_bbox_frame < bbox_lhws / 2)).all(dim=-1)
        assert mask.shape[-1:] == (n_pts,)
        mask = merge_masks(mask)
        # print('Number of points inside the boxes: ', mask.sum().item())
        # print('Total number of points: ', len(points))
        # print('Coverage ratio: ', mask.float().mean().item())

        return rewards, mask

    return rewards


def merge_rewards(rewards, eps=1e-6):
    assert isinstance(rewards, torch.Tensor)
    assert rewards.dim() >= 2
    n_pts = rewards.shape[-1]
    rewards = rewards.view(-1, n_pts)
    rewards = torch.clamp(rewards, eps, 1 - eps)
    lo = torch.log(1. - rewards)
    lo = lo.sum(dim=0)
    rewards = 1. - torch.exp(lo)
    # TODO: Stable logsumexp?
    # rewards = 1. - torch.logsumexp(1. - rewards, dim=(0, 1))
    assert rewards.shape == (n_pts,)
    return rewards


def merge_masks(masks):
    assert isinstance(masks, torch.Tensor)
    assert masks.dim() >= 2
    n_pts = masks.shape[-1]
    masks = masks.view(-1, n_pts)
    masks = masks.sum(dim=0)
    masks = masks > 0
    assert masks.shape == (n_pts,)
    return masks


def bboxes_iou(poses, sizes, eps=1e-3):
    assert isinstance(poses, torch.Tensor)
    assert isinstance(sizes, torch.Tensor)
    assert poses.shape[-2:] == (4, 4)
    assert sizes.shape[-1] == 3

    xyz_rpy = matrix_to_xyz_rpy(poses)
    assert xyz_rpy.shape[-1] == 6
    xyz_rpy_lhws = torch.cat([xyz_rpy, sizes], dim=-1)
    assert xyz_rpy_lhws.shape[-1] == 9
    xyz_rpy_lhws = xyz_rpy_lhws.view(-1, 9)

    # https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.box3d_overlap
    boxes = [Box(xyz_rpy_lhws[k, :3], xyz_rpy_lhws[k, 3:6], xyz_rpy_lhws[k, -3:]) for k in range(len(xyz_rpy_lhws))]
    verts = [box.get_verts() for box in boxes if box.get_volume() > eps]
    verts = torch.stack(verts, dim=0)
    verts = torch.as_tensor(verts, dtype=torch.float32)
    # TODO: ValueError: Plane vertices are not coplanar
    vol, iou = box3d_overlap(verts[:-1], verts[1:], eps=eps / 2.)

    return iou.mean()


def bboxes_optimization(points, inst_poses, inst_bbox_sizes, n_opt_iters=100, lr=0.1, grid_res=0.2, vis=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # put tensors to device
    points = points.to(device)
    inst_poses = inst_poses.to(device)
    inst_bbox_sizes = inst_bbox_sizes.to(device)

    # select instance trajectory and bounding boxes
    xyz_rpy = matrix_to_xyz_rpy(inst_poses)
    lhws = inst_bbox_sizes
    xyz_rpy = torch.as_tensor(xyz_rpy, dtype=points.dtype)
    lhws = torch.as_tensor(lhws, dtype=points.dtype)
    n_instances = len(xyz_rpy)

    # filter points approximately belonging to the trajectories
    masks = []
    for i in range(n_instances):
        L = trajlen(xyz_rpy[i]) + lhws[i].max()
        traj_center_pose = xyz_rpy_to_matrix(xyz_rpy[i].mean(dim=0))
        points_mask = filter_box(points,
                                 box_size=(L, L, L),
                                 box_T=traj_center_pose, only_mask=True)
        masks.append(points_mask[None])
    points_mask = merge_masks(torch.cat(masks, dim=0))
    points = points[points_mask]
    assert len(points) > 0, 'No points left after filtering. Are dynamic objects present in the scene?'
    # grid filter to reduce the number of points
    if grid_res is not None:
        points = filter_grid(points, grid_res=grid_res)

    # optimize bounding boxes
    xyz_yaw = xyz_rpy[..., [0, 1, 2, 5]]
    xyz_yaw.requires_grad = True
    optimizer = torch.optim.Adam([xyz_yaw], lr=lr)

    if vis:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(1500, 1000))
        frame_n = 0
    for i in range(n_opt_iters):
        poses = xyz_yaw_to_matrix(xyz_yaw)
        coverage, coverage_mask = bboxes_coverage(points.clone(), poses, lhws.clone(), sigmoid_slope=1., return_mask=True)
        path_cost = path_smoothness_batched(poses)
        loss = path_cost / (coverage + 1e-6)

        print('iter {}, loss: {:.4f}, coverage: {:.4f}, path_cost: {:.4f}'.format(i, loss.item(), coverage.item(), path_cost.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualize
        if vis and (i % 5 == 0 or i == n_opt_iters - 1):
            with torch.no_grad():
                mlab.clf()
                mlab.title('iter {}, loss: {:.4f}'.format(i, loss.item()), size=0.5, color=(0, 0, 0))
                draw_cloud(points[~coverage_mask].cpu(), color=(0, 0, 1), scale_factor=0.1)
                draw_cloud(points[coverage_mask].cpu(), color=(0, 1, 0), scale_factor=0.1)
                for i in range(len(poses.cpu())):
                    draw_coord_frames(poses.cpu()[i][::5], scale=0.5)
                    mlab.plot3d(poses.cpu()[i, :, 0, 3], poses.cpu()[i, :, 1, 3], poses.cpu()[i, :, 2, 3], color=(0, 0, 1), tube_radius=0.02)
                    # draw_bboxes(lhws.cpu()[i][::5], poses.cpu()[i][::5])

                # set up view point
                mlab.view(azimuth=0, elevation=180, distance=150.0)

                # # save frames
                # path = os.path.normpath('../gen/vis/bboxes_optimization')
                # os.makedirs(path, exist_ok=True)
                # mlab.savefig(os.path.join(path, 'frame_{:04d}.png'.format(frame_n)))
                # frame_n += 1

                fig.scene._lift()
    if vis:
        # # create video from the frames
        # os.system(f'ffmpeg -r 5 -i {path}/frame_%04d.png -vcodec mpeg4 -y ../gen/vis/bboxes_optimization.mp4')
        mlab.show()

    return xyz_yaw


def main():
    from data.dataloader import SFDataset4D

    # np.random.seed(0)
    # torch.manual_seed(0)

    n_time_stamps = 20
    min_trajlen = 2.0
    noise_std = 0.2
    n_opt_iters = 100
    lr = 0.1
    grid_res = 0.2

    # load dataset
    ds = SFDataset4D(dataset_type='waymo', n_frames=n_time_stamps)
    # data_i = np.random.randint(0, len(ds))
    data_i = 0
    print('Dataset index: ', data_i)
    data_sample = ds[data_i]

    # get instances trajectories and bounding boxes
    inst_bbox_sizes = get_inst_bbox_sizes(data_sample)
    ego_poses, inst_poses = get_inst_trajes(data_sample, min_traj_len=None, noise=None, axis_angle=False, verbose=False)

    # construct global cloud
    clouds = data_sample['pc1']
    points = clouds[0]
    for i in range(1, len(clouds)):
        cloud = clouds[i] @ ego_poses[i][:3, :3].T + ego_poses[i][:3, 3][None]
        points = torch.cat([points, cloud], dim=0)
    points = torch.as_tensor(points, dtype=inst_bbox_sizes[0].dtype)

    # discard short trajectories and the ones not observed for the whole time sequence
    traj_mask = [len(p) == n_time_stamps and trajlen(p) > min_trajlen for p in inst_poses]
    # assert torch.all(torch.tensor(traj_mask)),\
    #     'All trajectories should be of the same length and longer than min_trajlen. Try to decrease min_trajlen parameter.'
    inst_poses = torch.cat([p[None] for p, m in zip(inst_poses, traj_mask) if m], dim=0)
    inst_bbox_sizes = torch.cat([b[None] for b, m in zip(inst_bbox_sizes, traj_mask) if m], dim=0)

    n_instances = len(inst_poses)
    assert len(inst_poses) == len(inst_bbox_sizes) == n_instances
    assert inst_poses.shape == (n_instances, n_time_stamps, 4, 4)
    assert inst_bbox_sizes.shape == (n_instances, n_time_stamps, 3)

    # add noise
    if noise_std is not None:
        inst_poses[..., :3, 3] += torch.randn_like(inst_poses[..., :3, 3]) * noise_std

    # optimize bounding boxes
    xyz_yaw = bboxes_optimization(points, inst_poses, inst_bbox_sizes,
                                  n_opt_iters=n_opt_iters, lr=lr, grid_res=grid_res, vis=True)


if __name__ == '__main__':
    main()
