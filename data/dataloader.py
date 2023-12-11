import numpy as np
import torch
import glob
import importlib
from data.PATHS import DATA_PATH
from data.params.preprocessing_nsf import max_dist, min_dist, max_height, min_height
from data.path_utils import rel_poses2traj


def construct_global_poses_from_relative(relative_poses):
    ''' for pc1'''
    pose = torch.eye(4, device=relative_poses.device)[None]
    poses = pose.clone()
    poses12 = relative_poses
    for i in range(len(poses12)):
        pose = pose @ torch.linalg.inv(poses12[i])
        poses = torch.cat([poses, pose], dim=0)
    # poses = poses[1:]

    return poses

# if you want to use kiss-icp, just change the 'relative_poses' to output of icp
def compensate_ego_motion(data):
    '''Compensate ego motion for first frame - only ground truth so far, based it on the key in dictionary to swap to icp'''

    poses = construct_global_poses_from_relative(data['relative_pose'])
    pc1 = data['pc1'].clone()
    pc2 = data['pc2'].clone()
    gt_flow = data['gt_flow'].clone()
    compensated_gt_flow = gt_flow.clone()

    for i in range(len(pc1)):
        deformed_pc = (torch.cat([pc1[i], torch.ones_like(pc1[i][:, :1])], dim=1) @ data['relative_pose'][i].T.to(pc1.device))[None, :, :3]
        global_pc = (torch.cat([pc1[i], torch.ones_like(pc1[i][:, :1])], dim=1) @ poses[i].T.to(pc1.device))[None, :, :3]
        global_pc2 = (torch.cat([pc2[i], torch.ones_like(pc2[i][:, :1])], dim=1) @ poses[i].T.to(pc1.device))[None, :, :3]
        # gt flow must be compensated too
        flow_diff = deformed_pc - pc1[i:i + 1]
        pc1[i:i + 1] = global_pc  # print(pc1.shape, gt_flow.shape, flow_diff.shape, data['pc1'].shape)
        pc2[i:i + 1] = global_pc2
        compensated_gt_flow[i:i + 1, :, :3] = gt_flow[i:i + 1, :, :3] - flow_diff.to(pc1.device)  # for first frame
        # gt_flow[i:i+1,:,:3] = compensated_gt_flow

    data['global_pc1'] = pc1
    data['global_pc2'] = pc2
    data['compensated_gt_flow'] = compensated_gt_flow
    data['global_poses'] = poses

    return data

class NSF_dataset():
    # dataset type kittisf
    def __init__(self, root_dir=DATA_PATH, dataset_type: str = 'argoverse', subfold='val'):
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
        self.remap_func = dataset_module.remap_keys

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self.all_files):
            raise StopIteration
        # print(self.idx)
        # self.idx += 1
        return self.__getitem__(self.idx)

    def __getitem__(self, idx):
        # if self.idx == len(self.all_files):
        #     raise StopIteration

        data = np.load(self.all_files[self.idx])
        data = self.remap_func(data)

        pc1 = data['pc1']
        pc2 = data['pc2']
        gt_flow = data['gt_flow']

        pc1, pc2, gt_flow = self.preprocess_func(pc1, pc2, gt_flow)

        pc1 = torch.from_numpy(pc1).unsqueeze(0).to(torch.float32)
        pc2 = torch.from_numpy(pc2).unsqueeze(0).to(torch.float32)
        # pc_scene = torch.from_numpy(pc_scene).unsqueeze(0).to(torch.float32)

        pc1 = pc1 - self.lidar_pose
        pc2 = pc2 - self.lidar_pose

        gt_flow = torch.from_numpy(gt_flow).unsqueeze(0).to(torch.float32)

        self.idx += 1

        data['pc1'] = pc1
        data['pc2'] = pc2
        data['gt_flow'] = gt_flow

        return data

    def __len__(self):
        return len(self.all_files)


class SFDataset4D():

    def __init__(self, root_dir=DATA_PATH, dataset_type: str = 'argoverse', data_split='*', sequence='*', frame='*',
                 n_frames=1, only_first=False, **kwargs):

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.data_split = data_split
        self.sequence = sequence
        self.frame = frame
        self.only_first = only_first
        self.n_frames = n_frames
        self.idx = 0

        tmp_glob_path = f'{DATA_PATH}/{dataset_type}/processed/{data_split}/{sequence}/{frame}.npz'

        self.all_files = self.gather_all_frames(tmp_glob_path, self.n_frames)

        dataset_module = importlib.import_module('data.params.' + dataset_type)
        self.data_config = dataset_module.data_config

        # pre-process data, shift them if not in origin
        self.remap_keys = dataset_module.remap_keys
        # todo preprocessing step based on experiment cfg (function, not in params)
        # self.preprocess_func = dataset_module.frame_preprocess

        # indices = np.array(self.data_config['test_idx'], dtype=np.int32)
        # self.all_files = [dataset_module.all_files[idx] for idx in dataset_module.test_idx]

        # self.lidar_pose = self.data_config['lidar_pose']
        # self.lidar_pose = torch.tensor(self.data_config['lidar_pose']).unsqueeze(0).to(torch.float32)

    # def __iter__(self):
    #     return self

    def __getitem__(self, idx):

        # if self.idx == len(self.all_files):
        #     raise StopIteration

        # frame_path_list = self.get_adjacent_frames(self.all_files, self.idx, self.n_frames)
        frame_path_list = self.all_files[idx]
        # todo function is to load data based on params config
        # todo option to load only first n_frames per sequence

        # pc1, pc2, gt_flow = self.preprocess_func(pc1, pc2, gt_flow)
        data_npz_files = [np.load(data_path, allow_pickle=True) for data_path in frame_path_list]
        data_npz_files = [self.remap_keys(data_npz) for data_npz in data_npz_files]

        data_dict = self.collate_data_to_batch(data_npz_files)

        return data_dict

    def __len__(self):
        return len(self.all_files)

    def get_adjacent_frames(self, all_files, frame, n_frames):

        frame_path_list = [all_files[i] for i in range(frame, frame + n_frames) if
                           all_files[i].split('/')[-2] == all_files[frame].split('/')[-2]]

        return frame_path_list

    def gather_all_frames(self, glob_path, n_frames):

        all_files = sorted(glob.glob(glob_path))
        available_indices = []
        last_sequence = 'totally-not-a-sequence-name'

        for frame in range(len(all_files) - n_frames):

            frame_path_list = self.get_adjacent_frames(all_files, frame, n_frames)

            cur_sequence = frame_path_list[0].split('/')[-2]

            if self.only_first:

                if cur_sequence == last_sequence:
                    continue
                else:
                    last_sequence = cur_sequence

            # Drop if not all frames are available
            if len(frame_path_list) == n_frames:
                # available_indices.append(frame)
                available_indices.append(frame_path_list)

        # all_files = [all_files[i] for i in available_indices]
        all_files = available_indices

        return all_files

    def preprocess_func(self, pc_sample):

        pc = pc_sample - self.data_config['ground_origin']
        dist = np.linalg.norm(pc[:, :3], axis=1)
        mask = (dist < max_dist) & (dist > min_dist) & (pc[:, 2] < max_height) & (pc[:, 2] > min_height)

        return mask

    def collate_data_to_batch(self, data_npz_files):

        pc1_list = []
        pc2_list = []
        full_pc2_list = []
        gt_flow_list = []
        pose_list = []
        id_mask1_list = []
        mos1_list = []
        box1_list = []
        max_N = 0
        max_M = 0

        for data in data_npz_files:
            # Unpack
            pc1_sample = data['pc1'][:, :3]  # keep only geometry
            pc2_sample = data['pc2'][:, :3]  # keep only geometry
            full_pc2_sample = data['pc2'][:, :3]  # keep only geometry

            mask1 = self.preprocess_func(pc1_sample)
            mask2 = self.preprocess_func(pc2_sample)

            pc1_sample = pc1_sample[mask1]
            pc2_sample = pc2_sample[mask2]
            # print(data.keys())
            box1 = data['box1'] if self.dataset_type == 'waymo' else None
            gt_flow_sample = data['gt_flow'][mask1]
            pose_sample = data['relative_pose']
            id_mask1_sample = data['id_mask1'][mask1]
            mos1_sample = data['mos1'][mask1]

            # Calculate Maximum number of points
            max_N = np.max([max_N, pc1_sample.shape[0]])
            max_M = np.max([max_M, pc2_sample.shape[0]])

            pc1_list.append(pc1_sample)
            pc2_list.append(pc2_sample)
            full_pc2_list.append(full_pc2_sample)
            gt_flow_list.append(gt_flow_sample)
            pose_list.append(pose_sample)
            id_mask1_list.append(id_mask1_sample)
            mos1_list.append(mos1_sample)
            box1_list.append(box1)

        # Construct padded arrays
        pc1 = np.zeros((self.n_frames, max_N, 3))
        pc2 = np.zeros((self.n_frames, max_M, 3))

        gt_flow = np.zeros((self.n_frames, max_N, gt_flow_list[0].shape[-1]))
        id_mask1 = np.zeros((self.n_frames, max_N), dtype=int)
        mos1 = np.zeros((self.n_frames, max_N), dtype=bool)

        padded_mask_N = np.zeros((self.n_frames, max_N), dtype=bool)
        padded_mask_M = np.zeros((self.n_frames, max_M), dtype=bool)

        # Fill in the arrays with real points
        for i in range(len(pc1_list)):
            # point cloud 1
            pc1[i, :pc1_list[i].shape[0], :] = pc1_list[i]
            padded_mask_N[i, :pc1_list[i].shape[0]] = 1
            gt_flow[i, :gt_flow_list[i].shape[0], :] = gt_flow_list[i]
            id_mask1[i, :id_mask1_list[i].shape[0]] = id_mask1_list[i]
            mos1[i, :mos1_list[i].shape[0]] = mos1_list[i]

            # point cloud 2
            pc2[i, :pc2_list[i].shape[0], :] = pc2_list[i]
            padded_mask_M[i, :pc2.shape[0]] = 1

        # to Torch, device outside of this function as it shloud load everything at once (fastest)
        pc1 = torch.from_numpy(pc1).to(torch.float32)
        pc2 = torch.from_numpy(pc2).to(torch.float32)
        gt_flow = torch.from_numpy(gt_flow).to(torch.float32)
        padded_mask_N = torch.from_numpy(padded_mask_N).to(torch.bool)
        padded_mask_M = torch.from_numpy(padded_mask_M).to(torch.bool)
        pose = torch.from_numpy(np.stack(pose_list)).to(torch.float32)
        id_mask1 = torch.from_numpy(id_mask1).to(torch.int32)
        mos1 = torch.from_numpy(mos1).to(torch.bool)
        full_pc2 = [torch.from_numpy(p).to(torch.float32) for p in full_pc2_list]

        # print(gt_flow.shape, pc1.shape)
        data_dict = {'pc1': pc1, 'pc2': pc2, 'gt_flow': gt_flow, 'full_pc2' : full_pc2,
                     'padded_mask_N': padded_mask_N, 'padded_mask_M': padded_mask_M,
                     'relative_pose': pose, 'box1': box1_list,
                     'id_mask1': id_mask1, 'mos1': mos1,
                     }

        return data_dict


def global_cloud_demo():
    from vis.open3d import visualize_points3D
    from matplotlib import pyplot as plt
    from ops.filters import filter_grid

    ds = SFDataset4D(dataset_type='waymo', n_frames=40)
    # i = np.random.randint(len(ds))
    i = 0
    poses12 = ds[i]['relative_pose']
    # construct path from relative poses
    poses = rel_poses2traj(poses12)

    # generate global cloud
    clouds = ds[i]['pc1']
    global_cloud = clouds[0]
    for i in range(1, len(clouds)):
        cloud = clouds[i] @ poses[i][:3, :3].T + poses[i][:3, 3][None]
        global_cloud = torch.cat([global_cloud, cloud], dim=0)
    global_cloud = filter_grid(global_cloud, grid_res=0.5)
    visualize_points3D(global_cloud)

    plt.figure()
    plt.plot(poses[:, 0, 3], poses[:, 1, 3], 'o-')
    plt.axis('equal')
    plt.grid()
    plt.show()


def inst_poses_demo():
    import os
    from mayavi import mlab
    from vis.mayavi_interactive import draw_coord_frames
    from data.path_utils import get_inst_trajes

    ds = SFDataset4D(dataset_type='waymo', n_frames=40, only_first=True)
    ds_i = 0
    # ds_i = np.random.randint(len(ds))
    print('Dataset index: {}'.format(ds_i))
    data_sample = ds[ds_i]
    clouds = data_sample['pc1']

    ego_poses, inst_poses = get_inst_trajes(data_sample, min_traj_len=0.2, axis_angle=False, verbose=True)

    # construct global cloud
    global_cloud = clouds[0]
    for i in range(1, len(clouds)):
        cloud = clouds[i] @ ego_poses[i][:3, :3].T + ego_poses[i][:3, 3][None]
        global_cloud = torch.cat([global_cloud, cloud], dim=0)
    global_cloud = global_cloud[::10]

    # visualize scene in mayavi
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    mlab.plot3d(ego_poses[:, 0, 3], ego_poses[:, 1, 3], ego_poses[:, 2, 3], color=(0, 0, 1), tube_radius=0.1)
    draw_coord_frames(ego_poses[::5], scale=1.)
    for traj in inst_poses:
        color = tuple(np.random.random(3))
        mlab.plot3d(traj[:, 0, 3], traj[:, 1, 3], traj[:, 2, 3], color=color, tube_radius=0.1)
        mlab.points3d(traj[:, 0, 3], traj[:, 1, 3], traj[:, 2, 3], color=color, scale_factor=0.5)
    mlab.points3d(global_cloud[:, 0], global_cloud[:, 1], global_cloud[:, 2],
                  color=(0, 0, 0), scale_factor=0.1, opacity=0.1)
    mlab.show()


def bboxes_demo():
    import os
    from vis.mayavi_interactive import draw_coord_frames, draw_bboxes
    from data.path_utils import get_inst_trajes
    from data.box_utils import get_inst_bbox_sizes
    from data.PATHS import DATA_PATH
    from mayavi import mlab

    trajlen = lambda traj: torch.sqrt((torch.diff(traj[:, :3, 3] - traj[0, :3, 3], dim=0) ** 2).sum(dim=1)).sum() \
        if len(traj) > 1 else 0.

    ds = SFDataset4D(dataset_type='waymo', n_frames=20)
    data_sample = ds[0]

    inst_bbox_sizes = get_inst_bbox_sizes(data_sample)
    ego_poses, inst_poses = get_inst_trajes(data_sample, min_traj_len=None, noise=None, axis_angle=False)

    # construct global cloud
    clouds = data_sample['pc1']
    global_cloud = clouds[0]
    for i in range(1, len(clouds)):
        cloud = clouds[i] @ ego_poses[i][:3, :3].T + ego_poses[i][:3, 3][None]
        global_cloud = torch.cat([global_cloud, cloud], dim=0)

    # visualize traj in mayavi
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    for i in range(len(inst_poses)):
        poses = inst_poses[i]
        bboxes = inst_bbox_sizes[i]
        assert len(poses) == len(bboxes)

        if trajlen(poses) < 5.:
            # print(f'Trajectory of instance {i} has {len(poses)} poses and length: {trajlen(poses):.3f} [m]')
            continue

        mlab.plot3d(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], color=(0, 1, 0), tube_radius=0.01)
        draw_coord_frames(poses[::2], scale=0.5)
        draw_bboxes(bboxes[::2], poses[::2])
    mlab.points3d(global_cloud[::10, 0], global_cloud[::10, 1], global_cloud[::10, 2], color=(0, 0, 1), scale_factor=0.1)
    mlab.show()


def o3d_icp_demo():
    import open3d as o3d
    from vis.open3d import visualize_points3D

    ds = NSF_dataset(dataset_type='kitti_t')
    data = ds[0]

    pc1 = data['pc1'].squeeze()
    pc2 = data['pc2'].squeeze()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.estimate_normals()

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2 + np.asarray([0, 0, 1.]))
    pcd2.estimate_normals()

    # ICP registration
    threshold = 0.02
    Tr_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(pcd1, pcd2, threshold, Tr_init,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                                                      )
    print(reg)
    print("Transformation is:")
    print(reg.transformation)

    # visualize registration result
    pcd2.transform(reg.transformation)
    o3d.visualization.draw_geometries([pcd1, pcd2])


def icp_demo(grid_res=0.1, n_iters=500, lr=0.01, anim=False):
    from ops.filters import filter_grid
    from ops.transform import xyz_axis_angle_to_matrix
    from loss.icp import point_to_point_dist
    import open3d as o3d

    # load point clouds
    ds = NSF_dataset(dataset_type='kitti_t')
    data = ds[0]
    cloud1 = data['pc1'].squeeze()
    cloud2 = data['pc2'].squeeze()
    rel_pose_init = torch.eye(4)

    rel_pose = rel_pose_init.clone()
    cloud1_corr = cloud1.clone()

    # apply grid filtering to point clouds
    cloud1 = filter_grid(cloud1, grid_res=grid_res)
    cloud2 = filter_grid(cloud2, grid_res=grid_res)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cloud1 = torch.as_tensor(cloud1, dtype=torch.float32, device=device)
    cloud2 = torch.as_tensor(cloud2, dtype=torch.float32, device=device)
    rel_pose_init = torch.as_tensor(rel_pose_init, dtype=torch.float32, device=device)

    xyza1_delta = torch.zeros(6, device=device)
    xyza1_delta.requires_grad = True

    optimizer = torch.optim.Adam([{'params': xyza1_delta, 'lr': lr}])

    # open3d visualization
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2.detach().cpu())
    pcd2.paint_uniform_color([0, 0, 1])
    viewer.add_geometry(pcd1)
    viewer.add_geometry(pcd2)

    # run optimization loop
    for it in range(n_iters):
        # add noise to poses
        pose_deltas_mat = xyz_axis_angle_to_matrix(xyza1_delta[None]).squeeze()
        rel_pose = torch.matmul(rel_pose_init, pose_deltas_mat)

        # transform point clouds to the same world coordinate frame
        cloud1_corr = cloud1 @ rel_pose[:3, :3].T + rel_pose[:3, 3][None]

        loss = point_to_point_dist([cloud1_corr, cloud2], differentiable=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('At iter %i ICP loss: %f' % (it, loss))
        if anim:
            with torch.no_grad():
                # visualize in open3d
                pcd1.points = o3d.utility.Vector3dVector(cloud1_corr.detach().cpu())
                pcd1.paint_uniform_color([1, 0, 0])
                viewer.update_geometry(pcd1)
                viewer.poll_events()
                viewer.update_renderer()

    print('Final pose:\n', rel_pose.detach().cpu().numpy())

    # visualize in open3d
    pcd1.points = o3d.utility.Vector3dVector(cloud1_corr.detach().cpu())
    pcd1.paint_uniform_color([1, 0, 0])
    viewer.update_geometry(pcd1)
    viewer.run()
    viewer.destroy_window()

    # save result
    # np.savez('icp_result.npz', rel_pose=rel_pose.detach().cpu().numpy())
    return rel_pose.detach().cpu().numpy()


def model_output_demo():
    import open3d as o3d
    from vis.open3d import visualize_points3D
    from matplotlib import cm
    import matplotlib.pyplot as plt

    ds = NSF_dataset(dataset_type='kitti_t')
    # ds = SFDataset4D(dataset_type='waymo', n_frames=1, subfold='val')
    data = ds[0]

    out = np.load('/home/ruslan/CTU/sceneflow/experiments/multi-rigid-flow/33/inference/0000.npz', allow_pickle=True)

    cloud = data['pc1'].squeeze()
    red = np.repeat(np.array([[1, 0, 0]]), cloud.shape[0], axis=0)
    ids = out['id_mask1'].squeeze()
    # ids = data['id_mask1'].squeeze()

    cloud_next_gt = data['pc2'].squeeze()
    green = np.repeat(np.array([[0, 1, 0]]), cloud_next_gt.shape[0], axis=0)

    flow = data['gt_flow'].squeeze()
    # flow = out['pred_flow'].squeeze()

    cloud_pred = cloud + flow[..., :3]
    blue = np.repeat(np.array([[0, 0, 1]]), cloud_pred.shape[0], axis=0)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # for pcd, rgb in zip([cloud, cloud_next_gt, cloud_pred], [ids, red, green]):
    # for pcd, rgb in zip([cloud, cloud_pred], [red, blue]):
    for pcd, rgb in zip([cloud], [ids]):
        pcd = visualize_points3D(pcd, value=rgb, vis=False, colormap=cm.jet)
        vis.add_geometry(pcd)

    # set camera position
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=40.)
    vis.run()
    vis.destroy_window()

    # def draw_geometry_with_key_callback(pcd):
    #     def change_background_to_black(vis):
    #         opt = vis.get_render_option()
    #         opt.background_color = np.asarray([0, 0, 0])
    #         return False
    #
    #     def zoom_in(vis):
    #         ctr = vis.get_view_control()
    #         ctr.scale(1.25)
    #         return False
    #
    #     def load_render_option(vis):
    #         vis.get_render_option().load_from_json(
    #             "../../TestData/renderoption.json")
    #         return False
    #
    #     def capture_depth(vis):
    #         depth = vis.capture_depth_float_buffer()
    #         plt.imshow(np.asarray(depth))
    #         plt.show()
    #         return False
    #
    #     def capture_image(vis):
    #         image = vis.capture_screen_float_buffer()
    #         plt.imshow(np.asarray(image))
    #         plt.show()
    #         return False
    #
    #     key_to_callback = {}
    #     key_to_callback[ord("K")] = change_background_to_black
    #     key_to_callback[ord("R")] = load_render_option
    #     key_to_callback[ord("Z")] = zoom_in
    #     key_to_callback[ord(",")] = capture_depth
    #     key_to_callback[ord(".")] = capture_image
    #     o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    #
    # pcd = visualize_points3D(cloud, value=ids, vis=False, colormap=cm.jet)
    # # draw_geometry_with_custom_fov(pcd, 90.)
    # # draw_geometry_with_rotation(pcd)
    # draw_geometry_with_key_callback(pcd)


def demo():
    import open3d as o3d
    from vis.open3d import visualize_points3D
    from ops.filters import filter_grid, filter_box

    def transform_cloud(cloud, pose):
        return cloud @ pose[:3, :3].T + pose[:3, 3][None]

    ds = SFDataset4D(dataset_type='waymo', n_frames=40)
    i = 20
    poses12 = ds[i]['relative_pose']
    # construct path from relative poses
    poses = rel_poses2traj(poses12)

    # generate global cloud
    clouds = ds[i]['pc1']
    cloud1 = transform_cloud(clouds[0], poses[1])
    cloud2 = transform_cloud(clouds[1], poses[2])

    def draw_geometry_with_key_callback(geoms):
        def set_viewpoint(vis):
            ctr = vis.get_view_control()
            params = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
            ctr.convert_from_pinhole_camera_parameters(params)
            return False

        def save_viewpoint(vis):
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('viewpoint.json', params)
            return False

        key_to_callback = {}
        key_to_callback[ord("A")] = set_viewpoint
        key_to_callback[ord("S")] = save_viewpoint
        o3d.visualization.draw_geometries_with_key_callbacks(geoms, key_to_callback)

    # visualize clouds
    pcd1 = visualize_points3D(cloud1, value=None, vis=False)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2 = visualize_points3D(cloud2, value=None, vis=False)
    pcd2.paint_uniform_color([0, 0, 1])

    geoms = [pcd1, pcd2]

    draw_geometry_with_key_callback(geoms)


def data_statistics():
    from tqdm import tqdm
    import pandas as pd

    data_names = ['kittisf', 'argoverse', 'waymo', 'nuscenes']
    stats = pd.DataFrame(columns=['name', 'n_samples', 'n_pts', 'ratio_dyn_stat'])

    for name in data_names:
        if name in ['kittisf']:
            DS = NSF_dataset
            dyn_mask_field = 'id_mask1'
        else:
            DS = SFDataset4D
            dyn_mask_field = 'mos1'
        cloud_field = 'pc1'

        ds = DS(dataset_type=name)
        if len(ds) == 0:
            print(f'No data samples in {name} found')
            continue
        print(f'Number of samples in {name}: {len(ds)}')
        # continue

        average_n_pts = 0
        n_dynamic_pts = 0
        n_static_pts = 0
        for i, data in enumerate(tqdm(ds)):
            # if i > 10: break
            assert cloud_field in data.keys()
            assert dyn_mask_field in data.keys()
            pts = data[cloud_field].squeeze()
            average_n_pts += pts.shape[0]
            mask_dyn = data[dyn_mask_field].squeeze()
        
            if i == 0:
                #print(f'{name} contains data fields:', data.keys())
                #print(pts.shape, mask_dyn.shape)
                #print(np.unique(mask_dyn))
                pass

            assert mask_dyn.shape[0] == pts.shape[0]
            n_dynamic_pts += (mask_dyn > 0).sum()
            n_static_pts += (mask_dyn == 0).sum()
        average_n_pts /= len(ds)
        average_n_pts = int(average_n_pts)
        print(f'Average number of points in {name}: {average_n_pts}')

        ratio_dyn_stat = float(n_dynamic_pts / n_static_pts) if n_static_pts > 0. else 0.
        print(f'Ratio of dynamic to static points in {name}: {ratio_dyn_stat}')
        
        new_row = {'name': name, 'n_samples': len(ds), 'n_pts': average_n_pts, 'ratio_dyn_stat': ratio_dyn_stat}
        stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
    
    # save stats to csv
    stats.to_csv('data_statistics.csv', index=False)


def main():
    # demo()
    # model_output_demo()
    # global_cloud_demo()
    # inst_poses_demo()
    # bboxes_demo()
    # icp_demo(lr=0.002, anim=True)
    data_statistics()


if __name__ == '__main__':
    main()
