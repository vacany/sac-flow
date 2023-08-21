import os
import numpy as np
import torch
from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib
def visualize_flow_frame(pc1, pc2, est_flow1, fig=1):

    figure = mlab.figure(fig, bgcolor=(1, 1, 1), size=(1024, 1024))
    if type(pc1) == torch.Tensor:
        vis_pc1 = pc1.detach().cpu().numpy()
        vis_pc2 = pc2.detach().cpu().numpy()
        vis_est_rigid_flow = est_flow1.detach().cpu().numpy()

    else:
        vis_pc1 = pc1
        vis_pc2 = pc2
        vis_est_rigid_flow = est_flow1

    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')

    mlab.points3d(vis_pc1[:,0], vis_pc1[:,1], vis_pc1[:,2], color=(0,0,1), scale_factor=0.1)
    mlab.points3d(vis_pc2[:,0], vis_pc2[:,1], vis_pc2[:,2], color=(1,0,0), scale_factor=0.1)
    mlab.quiver3d(vis_pc1[:,0], vis_pc1[:,1], vis_pc1[:,2], vis_est_rigid_flow[:,0], vis_est_rigid_flow[:,1], vis_est_rigid_flow[:,2], color=(0,1,0), scale_factor=1)

    # if pose2 is not None:
        # mlab.points3d(pose2.detach()[0, 0, -1], pose2.detach()[0, 1, -1], pose2.detach()[0, 2, -1], color=(1, 0, 0), scale_factor=0.3, mode='axes')
    # if normals1 is not None:
    #     vis_normals1 = normals1.detach().cpu().numpy()
    #     mlab.quiver3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], vis_normals1[0, :, 0], vis_normals1[0, :, 1], vis_normals1[0, :, 2], color=(0, 0, 0.4), scale_factor=0.4)
    #
    # if normals2 is not None:
    #     vis_normals2 = normals2.detach().cpu().numpy()
    #     mlab.quiver3d(vis_pc2[0,:,0], vis_pc2[0,:,1], vis_pc2[0,:,2], vis_normals2[0, :,0], vis_normals2[0, :,1], vis_normals2[0, :,2], color=(0.4,0,0), scale_factor=0.4)

def visualize_flow_points(pc1, pc2, est_flow, fig=1):
    figure = mlab.figure(fig, bgcolor=(1, 1, 1), size=(1024, 1024))
    mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0, 0, 1), scale_factor=0.1)
    # mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(1, 0, 0), scale_factor=0.1)
    mlab.points3d(pc1[:, 0] + est_flow[:,0], pc1[:, 1] + est_flow[:,1], pc1[:, 2] + est_flow[:,2], color=(0, 1, 0), scale_factor=0.1)

# def visualize_flow_error(pc1, pc2, est_flow, gt_flow):

def visualize_flow_error(pc1, est_flow, gt_flow, cmap='jet', vmin=0, vmax=np.inf, fig=1):

    s = np.arange(len(pc1))
    error = np.linalg.norm(est_flow - gt_flow, axis=1)
    error[0] = 1.8    # to make the colorbar consistent for both baseline and our method
    error[1] = 0.001
    # error[1] = 0.000043
    # error -= error.min()
    # error = error > 0.3
    print('Figure: ', fig, 'Errors:', error.min(), error.max())
    np.log(15. * error, out=error, where=error > 0) # log scale
    color_mapping = matplotlib.cm.get_cmap(cmap)
    # vmax = 2
    lut = (color_mapping(np.clip(error, vmin, vmax)) * 255).astype(np.uint8)

    figure = mlab.figure(fig, bgcolor=(1, 1, 1), size=(1024, 1024))
    # Plot the points, update its lookup table
    p3d = mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], s, scale_mode='none', scale_factor=0.1,
                        mode='sphere', colormap='jet')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.colorbar(object=p3d, title="L1 Error")
    mlab.draw()

def visualize_frame(frame_path, flow_type='est_flow', mode='arrows', fig=1):

    data_npz = np.load(frame_path, allow_pickle=True)

    pc1 = data_npz['pc1']
    pc2 = data_npz['pc2']
    est_flow = data_npz[flow_type]
    gt_flow = data_npz['gt_flow']
    # epe_all = data_npz['epe_all']
    if mode == 'arrows':
        visualize_flow_frame(pc1, pc2, est_flow, fig)
    elif mode == 'points':
        visualize_flow_points(pc1, pc2, est_flow, fig)
    elif mode == 'error':
        visualize_flow_error(pc1, est_flow, gt_flow, fig=fig)

def visualize_difference(ours_path, base_path, fig=1):
    ours_data_npz = np.load(ours_path, allow_pickle=True)
    base_data_npz = np.load(base_path, allow_pickle=True)

    pc1 = base_data_npz['pc1']
    pc2 = base_data_npz['pc2']
    est_flow = base_data_npz['est_flow'] - ours_data_npz['est_flow']

    mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=np.linalg.norm(est_flow,axis=1), scale_factor=0.1)
    # visualize_flow_frame(pc1, pc2, est_flow, fig=fig)

def generate_kitti_t_images():
    our_frame_path = '/home/patrik/rci/experiments/normals_5/2023-08-14-13-48-38-499/inference/000000.npz'
    base_frame_path = '/home/patrik/rci/experiments/normals_0/2023-08-14-13-51-20-694/inference/000000.npz'

    visualize_frame(our_frame_path, flow_type='est_flow', mode='arrows', fig=1)
    # view = (158.5562734972502, 20.677887965391545, 4.474007709522948, np.array([17.04637354,  2.15330311,  1.30967617]))
    # view = (158.55627349725026, 20.677887965391545, 7.925977571890188, np.array([17.79594321,  1.7848886 ,  1.6238355 ]))
    view = (113.17076974835378, 56.84032751911419, 5.413549328522778, np.array([18.9776263 ,  3.05840809,  0.42642426]))

    mlab.view(*view)
    # mlab.show()
    mlab.savefig('paper/img/normals_ablation_our.png')
    # mlab.close()


    visualize_frame(base_frame_path, flow_type='est_flow', mode='arrows', fig=2)
    mlab.view(*view)
    mlab.savefig('paper/img/normals_ablation_base.png')
    # mlab.close()

    visualize_frame(our_frame_path, flow_type='gt_flow', mode='arrows', fig=3)
    mlab.view(*view)
    mlab.savefig('paper/img/normals_ablation_gt.png')
    # mlab.close()

def generate_argoverse_images(frame):

    our_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_ours/2023-08-02-12-08-35-038/inference/{frame:06d}.npz'
    base_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_base/2023-08-02-12-08-04-119/inference/{frame:06d}.npz'

    # mode = 'points'
    # mode = 'arrows'
    # frame_182_view = (43.810038233599315, 55.01040449787942, 25.56012250151007, np.array([-12.12779955, -11.14639799, -6.84791685]))
    frame_182_view = (26.81113283596679, 67.25882492023514, 21.12406818306622, np.array([-14.47408668,  -9.26251115,  -4.84168552]))


    for mode in ['error']:

        visualize_frame(our_frame_path, flow_type='est_flow', mode=mode, fig=1)
        mlab.view(*frame_182_view)
        mlab.savefig(f'/home/patrik/pcflow/paper/img/new_argoverse_points/argoverse_ours_{mode}.png')
        # mlab.show()
        mlab.close()

        visualize_frame(base_frame_path, flow_type='est_flow', mode=mode, fig=2)
        mlab.view(*frame_182_view)
        # mlab.show()
        mlab.savefig(f'/home/patrik/pcflow/paper/img/new_argoverse_points/argoverse_base_{mode}.png')
        mlab.close()

        # visualize_frame(our_frame_path, flow_type='gt_flow', mode=mode, fig=3)
        # mlab.view(*frame_182_view)
        # # mlab.show()
        # mlab.savefig(f'/home/patrik/pcflow/paper/img/new_argoverse_points/argoverse_gt_{mode}.png')
        # mlab.close()



    # visualize_difference(our_frame_path, base_frame_path, fig=1)
    # mlab.show()

# generate_argoverse_images()


if __name__ == '__main__':
    # frame = 3
    # for frame in range(195):
    #     our_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_ours/2023-08-02-12-08-35-038/inference/{frame:06d}.npz'
    #     base_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_base/2023-08-02-12-08-04-119/inference/{frame:06d}.npz'
    #
    #     our = np.load(our_frame_path, allow_pickle=True)
    #     base = np.load(base_frame_path, allow_pickle=True)
    #
    #     our_epe = our['EPE3D']
    #     base_epe = base["EPE3D"]
    #
    #     print(f'Frame: {frame:03d} \t ours:  {our_epe:.3f} \t base: {base_epe:.3f}')
    #

    from vis.deprecated_vis import visualize_points3D
    import sys

    # frame = int(sys.argv[1])
    # frame = 178
    # our_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_ours/2023-08-03-08-26-32-733/inference/{frame:06d}.npz'
    # base_frame_path = f'/home/patrik/rci/experiments/argoverse_qualitative_base/2023-08-03-08-26-33-979/inference/{frame:06d}.npz'
    # our = np.load(our_frame_path, allow_pickle=True)
    # base = np.load(base_frame_path, allow_pickle=True)
    # our_error = np.linalg.norm(our['est_flow'] - our['gt_flow'], axis=1)
    # base_error = np.linalg.norm(base['est_flow'] - our['gt_flow'], axis=1)
    #
    # generate_kitti_t_images()

    # visualize_points3D(our['pc1'], np.clip(base_error, 0, 1), bg_color=(1,1,1,0), show_grid=False, point_size=0.02, show_axis=False)
    # visualize_points3D(our['pc1'], np.clip(our_error, 0, 1), bg_color=(1,1,1,0), show_grid=False, point_size=0.02, show_axis=False)

    # generate_argoverse_images(178)
    pc1 = np.random.rand(100,3)
    s = np.arange(len(pc1))
    error = np.random.randint(0, 20, size=(100,)) / 20

    cmap = 'jet'
    vmin = 0
    vmax = np.inf
    fig = 1

    color_mapping = matplotlib.cm.get_cmap(cmap)
    # vmax = 2
    lut = (color_mapping(np.clip(error, vmin, vmax)) * 255).astype(np.uint8)

    figure = mlab.figure(fig, bgcolor=(1, 1, 1), size=(1024, 1024))
    # Plot the points, update its lookup table
    p3d = mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], s, scale_mode='none', scale_factor=0.1,
                        mode='sphere', colormap='jet')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    mlab.show()
    # normals_file = np.load('/home/patrik/rci/pcflow/normals.npz', allow_pickle=True)
    # pc1 = normals_file['pc']
    # normal = normals_file['normals_K3']
    #
    # mlab.figure(1,bgcolor=(0.1,0.1,0.1), size=(1920, 1080))
    # mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], color=(0,0,1), scale_factor=0.1)
    # mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], normal[:,0], normal[:,1], normal[:,2], color=(0.9,0.9,0.9), scale_factor=0.4)
    # mlab.show()
    #
    # normal = normals_file['normals_K4']
    # mlab.figure(2, bgcolor=(0.1, 0.1, 0.1), size=(1920, 1080))
    # mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0, 0, 1), scale_factor=0.1)
    # mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], color=(0.9, 0.9, 0.9),
    #               scale_factor=0.4)
    # mlab.show()
