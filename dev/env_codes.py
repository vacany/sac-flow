from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals

import mayavi.mlab as mlab
from ops.transform import *
from loss.flow import smoothness_loss

#
# if 'pycharm' in sys.argv[0]:
#     frame = 162
#     path = f'/home/patrik/rci/experiments/scoop_K32_normals/pc_res/{frame:06d}_res.npz'
# else:
#     frame = int(sys.argv[1])
#     path = f'/home/patrik/rci/experiments/scoop_{sys.argv[2]}/pc_res/{frame:06d}_res.npz'
#
# data = np.load(path, allow_pickle=True)
#
# print(data.files)



N = 100
pc1 = torch.rand(1, N, 3) + 4
gt_rigid_flow = torch.rand(1, 1, 3) * 3
# gt_rigid_flow.requires_grad = True

gt_dynamic_flow = torch.zeros(pc1.shape)
gt_dynamic_flow[:, 0, :] = torch.rand(1, 3) * 5.
full_flow = gt_rigid_flow + gt_dynamic_flow

pc2 = pc1 + gt_rigid_flow + gt_dynamic_flow
est_flow1 = torch.rand(1, N, 3, requires_grad=True)

weights = torch.ones(1, pc1.shape[1])
optimizer = torch.optim.Adam([est_flow1], lr=0.1)

smooth_dist, smooth_nn_idx, _ = knn_points(pc1, pc1, K=12)
# smooth_nn_idx.requires_grad = False

normals1 = estimate_pointcloud_normals(pc1, neighborhood_size=3)
normals2 = estimate_pointcloud_normals(pc2, neighborhood_size=3)
### loop

for i in range(2000):

    chamf_dist, nn_ind, _ = knn_points(pc1+est_flow1, pc2, lengths1=None, lengths2=None, K=1) # return_nn will come in handy later
    flow_normals = estimate_pointcloud_normals(pc1+est_flow1, neighborhood_size=3)
    normals_dist = torch.nn.functional.mse_loss(flow_normals, normals2)
    smooth_loss, smooth_per_point = smoothness_loss(est_flow1, NN_idx=smooth_nn_idx[0])
    loss = + chamf_dist.mean() + 0.8 * normals_dist.mean() + smooth_loss.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print('frame ', i, loss.item())



pose = find_weighted_rigid_alignment(pc1, pc1 + est_flow1, weights=weights)
d1 = pc1
d2 = pc2

# equation triangle
move_threshold = 0.05
diff_from_static = d1 + pose[:,:3,-1] - d2  # synthetic data

potentially_dynamic = diff_from_static.norm(dim=2) > move_threshold


mlab.figure(1)
stat_pc2 = pc2[potentially_dynamic==False]
dyn_pc2 = pc2[potentially_dynamic==True]
mos_mask = potentially_dynamic==True

# color_mask = np.array(((0,0,1), (1,0,0)))[potentially_dynamic.to(torch.long)]

# ground grid
def create_ground_surface(min_x=-10, max_x=10, min_y=-10, max_y=10, step=1.):
    x_gr = torch.arange(min_x, max_y, step)
    y_gr = torch.arange(min_y, max_y, step)
    grid_x, grid_y = torch.meshgrid(x_gr, y_gr)

    gr_z = torch.zeros_like(grid_x.flatten())

    gr_xyz = torch.stack((grid_x.flatten(), grid_y.flatten(), gr_z), dim=1)

    ground_normals = estimate_pointcloud_normals(gr_xyz.unsqueeze(0), neighborhood_size=8)
    # in criterion, make the z-normal face upwards

    # add flow as well?
    ground_surface = torch.cat((gr_xyz, ground_normals[0]), dim=1)


    return ground_surface

# optimize height of ground_surface
gr_xyz = create_ground_surface(step=1).unsqueeze(0)

z_surf = torch.full_like(gr_xyz[:,:,:1], fill_value=pc2[...,2].min(), requires_grad=True)


ground_surf = torch.cat((gr_xyz[:,:,:2], z_surf), dim=2)
gr_optimizer = torch.optim.Adam([z_surf], lr=0.5)

# find lowest points in the pointcloud based on 2d projection
# orig_raster = gr_xyz[:,:, :2].detach().clone()

_, gr_smooth_nn_ind, _ = knn_points(ground_surf, ground_surf, K=5)

# ground anchor points from local smoothness as minimal height from pc2?
for i in range(100):

    gr_lenght = torch.tensor(gr_xyz.size()[1]).unsqueeze(0)
    pc2_lenght = torch.tensor(pc2.size()[1]).unsqueeze(0)
    gr_dist, gr_nn_ind, _ = knn_points(ground_surf[:,:,:2], pc2[:,:,:2], lengths1=gr_lenght, lengths2=pc2_lenght, K=1)

    # xy distance - it still does not respect the lowest point, it needs to be formally defined?
    occ_ground_mask = gr_dist < 0.5
    masked_gr_nn_ind = gr_nn_ind[occ_ground_mask]

    matched_pts = pc2[:, masked_gr_nn_ind]
    masked_z_surf = z_surf[occ_ground_mask].unsqueeze(0).unsqueeze(2)

    gr_z_dist = torch.nn.functional.mse_loss(masked_z_surf, matched_pts[..., 2:3])

    # normal smoothness
    gr_normals = estimate_pointcloud_normals(ground_surf, neighborhood_size=4)
    gr_normals_loss = torch.nn.functional.mse_loss(gr_normals[..., 2], torch.ones_like(gr_normals[..., 2]))


    gr_smooth_loss, gr_smooth_per_point = smoothness_loss(z_surf, NN_idx=gr_smooth_nn_ind[0])
    # shape_loss = torch.nn.functional.mse_loss(gr_xyz[:,:,:2], orig_raster)

    # if minimizing all NN, initialize from min of point cloud and including smoothness and normals, it is stuced in beginning?

    ground_loss = gr_z_dist.mean() + 10 * gr_smooth_loss + gr_normals_loss
    ground_loss.backward()
    gr_optimizer.step()
    gr_optimizer.zero_grad()

    print('ground ', i, ground_loss.item())

ground_surf[0,:,2] = z_surf[0,:,0].detach()

mlab.points3d(stat_pc2[:,0], stat_pc2[:,1], stat_pc2[:,2], color=(1,1,0), scale_factor=0.1)
mlab.points3d(dyn_pc2[:,0], dyn_pc2[:,1], dyn_pc2[:,2], color=(1,0,1), scale_factor=0.1)
mlab.points3d(ground_surf[0,:,0].detach(), ground_surf[0,:,1].detach(), ground_surf[0,:,2].detach(), color=(0,0,0), scale_factor=0.1)
mlab.quiver3d(ground_surf[0,:,0].detach(), ground_surf[0,:,1].detach(), ground_surf[0,:,2].detach(), gr_normals[0,:,0].detach(), gr_normals[0,:,1].detach(), gr_normals[0,:,2].detach(), color=(0,0,0), scale_factor=0.5)


# Derivatives not verified!
derivative_z_surf_x = torch.gradient(z_surf.reshape(1,20,20,1), dim=1)[0]
derivative_z_surf_y = torch.gradient(z_surf.reshape(1,20,20,1), dim=2)[0]
derivative_z_surf = torch.sqrt(derivative_z_surf_x.reshape(1,400,1)**2 + derivative_z_surf_y.reshape(1,400,1)**2)

mlab.points3d(ground_surf[0,:,0].detach(), ground_surf[0,:,1].detach(), derivative_z_surf[0,:,0].detach(), color=(0.8,0.8,0.8), scale_factor=0.1)

mlab.show()





from vis import *

visualize_flow_frame(pc1, pc2, est_flow1, pose2=pose, normals1=normals1, normals2=normals2)
# visualize_flow_frame(torch.zeros_like(pc1), pc2, d1, pose2=pose, normals1=None, normals2=None)
# visualize_flow_frame(torch.ones_like(pc1) * pose[:, :3, -1], pc2, d2, pose2=pose, normals1=None, normals2=None)

# todo rewrite to losses that takes input args, and if none, then it calculates them itself
# todo ground to functions
# todo vis to file
