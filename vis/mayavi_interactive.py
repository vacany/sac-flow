import numpy as np
from mayavi import mlab

def visualize_PCA(pc, eigen_vectors):
    figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
    vis_pc = pc.detach().cpu().numpy()
    vis_eigen_vectors = eigen_vectors.detach().cpu().numpy()


    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    # mlab.points3d(vis_pc[0,:,0], vis_pc[0,:,1], vis_pc[0,:,2], color=(0,0,1), scale_factor=0.1)
    mlab.quiver3d(np.zeros(3), np.zeros(3), np.zeros(3), vis_eigen_vectors[:,0], vis_eigen_vectors[:,1], vis_eigen_vectors[:,2], color=(0,1,0), scale_factor=1)

    mlab.show()

def visualize_flow_frame(pc1, pc2, est_flow1, pose=None, pose2=None, normals1=None, normals2=None):

    figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
    vis_pc1 = pc1.detach().cpu().numpy()
    vis_pc2 = pc2.detach().cpu().numpy()

    vis_est_rigid_flow = est_flow1.detach().cpu().numpy()

    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    if pose2 is not None:
        mlab.points3d(pose2.detach()[0, 0, -1], pose2.detach()[0, 1, -1], pose2.detach()[0, 2, -1], color=(1, 0, 0), scale_factor=0.3, mode='axes')
    mlab.points3d(vis_pc1[0,:,0], vis_pc1[0,:,1], vis_pc1[0,:,2], color=(0,0,1), scale_factor=0.1)
    mlab.points3d(vis_pc2[0,:,0], vis_pc2[0,:,1], vis_pc2[0,:,2], color=(1,0,0), scale_factor=0.1)
    mlab.quiver3d(vis_pc1[0,:,0], vis_pc1[0,:,1], vis_pc1[0,:,2], vis_est_rigid_flow[0, :,0], vis_est_rigid_flow[0, :,1], vis_est_rigid_flow[0, :,2], color=(0,1,0), scale_factor=1)

    if normals1 is not None:
        vis_normals1 = normals1.detach().cpu().numpy()
        mlab.quiver3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], vis_normals1[0, :, 0], vis_normals1[0, :, 1], vis_normals1[0, :, 2], color=(0, 0, 0.4), scale_factor=0.4)

    if normals2 is not None:
        vis_normals2 = normals2.detach().cpu().numpy()
        mlab.quiver3d(vis_pc2[0,:,0], vis_pc2[0,:,1], vis_pc2[0,:,2], vis_normals2[0, :,0], vis_normals2[0, :,1], vis_normals2[0, :,2], color=(0.4,0,0), scale_factor=0.4)
    mlab.show()
