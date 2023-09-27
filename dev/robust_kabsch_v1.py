import torch
import torch.nn as nn
import numpy as np
from data.PATHS import DATA_PATH
from loss.flow import *

from vis.deprecated_vis import *
from models.simple_models import InstanceSegModule, FlowSegPrior, Weights_model, JointFlowInst
import matplotlib.pyplot as plt

from loss.flow import chamfer_distance_loss, SmoothnessLoss
from loss.instance import DynamicLoss, InstanceSmoothnessLoss
from loss.utils import find_robust_weighted_rigid_alignment
from loss.flow import DT

from data.NSF_data import NSF_dataset

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

dataset = NSF_dataset(dataset_type='kitti_t')
batch = dataset.__next__()
pc1 = batch['pc1'].to(device)
pc2 = batch['pc2'].to(device)
gt_flow = batch['gt_flow'].to(device)


# Model
model = Weights_model()
model.update(pc1, pc2)  # THIS CAN MAKE NON GRAD WITH OPTIMIZER
model.to(device)
# model = InstanceSegRefiner(pc1, max_instances=4)
# model = InstanceSegPrior(instances=10, layer_size=12).to(device)
# model = JointFlowInst().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

K = 8
max_radius = 1
grid_factor = 10

# losses
dt = DT(pc1, pc2.clone().to(pc1.device), grid_factor)
Smoothness_loss = SmoothnessLoss(pc1=pc1, pc2=pc2, K=8, max_radius=1, forward_weight=1, sm_normals_K=0, pc2_smooth=True, dataset='argoverse')
IS_smooth_loss = InstanceSmoothnessLoss(pc1, K=K, max_radius=max_radius)
Dynamic_loss = DynamicLoss(loss_norm=2)




for flow_e in range(1000):
    # todo
    # [ ] integrate to original codebase
    # [ ] Instances from ground truth, set up losses, if it helps
    # [ ] Code it on KITTISF (prepared instances), set up code base with instances and visuals in 2D
    # [ ] Visuals focused on instances with bad flow
    # [x] replicate ogc
    # [x] do it separately first?
    pc1 = pc1.contiguous()
    mask, pred_flow = model(pc1)

    # pred_flow = out[..., :3]
    # mask = out[..., 3:].softmax(dim=2)

    # deformed_pc1 = tr_pc1 @ translation
    # pred_flow = deformed_pc1[..., :3] - pc1
    # loss_flow, _, _ = chamfer_distance_loss(pc1 + pred_flow, pc2)
    loss_flow, _ = dt.torch_bilinear_distance(pc1 + pred_flow)
    cycle_smooth_flow = Smoothness_loss(pc1, pred_flow, pc2)

    # pred_flow = pred_flow.detach()
    # pred_flow = pred_flow

    # Instance Segmentation
    # mask = model(pc1)

    # mask = torch.softmax(mask, dim=2)

    w = mask[:, :, 0]  # how to init to calculate Kabsch from all in the beginning? --- Gumpel softmax trick
    #
    # # Gumpel softmax trickred_flow)
    w_hard = (mask.max(dim=2, keepdim=True)[1] == 0).to(torch.float)[:,:,0]
    kabsch_w = w_hard - w.detach() + w
    #
    # # here can switch gt and pred flow
    transform = find_robust_weighted_rigid_alignment(pc1, pc1 + pred_flow, weights=kabsch_w)    # Gumpel is not necessary maybe?
    #
    to_transform_pc1 = torch.cat((pc1, torch.ones_like(pc1[..., :1])), dim=2)
    # # FAKIN Transformace!!!
    sync_pc1 = torch.bmm(to_transform_pc1, transform.transpose(2,1))[..., :3]
    #
    # here can switch gt and pred flow
    difference = sync_pc1 - (pc1 + pred_flow)
    rmsd = difference.norm(dim=2)

    # pseudo_label
    # static_pseudo_label = rmsd < 0.05

    # Smoothness IS
    smooth_loss, per_point_smooth_loss = IS_smooth_loss(mask)
    dynamic_loss = Dynamic_loss(pc1, mask, pred_flow)
    # loss = rmsd.mean() + smooth_loss.mean() + 0.1 * (w * rmsd).mean()


    # loss = + loss_flow #+ rmsd.mean() #+ 0.1 * (w * rmsd).mean()
    loss =  loss_flow + smooth_loss.mean() + cycle_smooth_flow + dynamic_loss + w * rmsd.mean()# + 0.1 * (w * rmsd).mean()

    # Logic in Loss:
    # rmsd ---> decrease background point instances until rmsd is zero (static flow only, needs to be correct of course)
    # smoothness ---> merge nearby instances
    # w * rmsd ---> use residual rmsd to decrease background and increase foreground instances - Rigid trans might be okay with 1) loss and this transform everything else to dynamic

    # notes:
    # - use 3rd term for instance segmentation clustering

    # rmsd ---> dynamic
    # Refine instances based on rmds?

    loss.mean().backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Iter: {flow_e:03d} \t Loss: {loss.mean().item():.4f} \t Flow: {loss_flow.mean().item():.4f} \t Smoothness: {smooth_loss.mean().item():.4f} "
          # f"Cycle: {cycle_smooth_flow.mean().item():.4f} \t Dynamic: {dynamic_loss.mean().item():.4f} \t"
          # f"RMSD: {rmsd.mean().item():.4f} \t Kabsch_w: {kabsch_w.mean().item():.4f}"
          )

# TODO:
# flying things, metrics, name of dataset
# write visualizer module from this scripts
# odstranit margin
# metriky, flow na testovacich datech blbe
# flow na trenovacich blbe - loss funkce blbe?
# visualizer module, that visualizes losses as well
# always present, but flag turn it of
# inherits loss module at current state as well
# dev mode for fast search in visuals, iterations keep for last (maybe not needed)
# read https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Neural_Prior_for_Trajectory_Estimation_CVPR_2022_paper.pdf
#


visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), pred_flow[0].detach().cpu().numpy())
# visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), gt_flow[0].detach().cpu().numpy())

# visualize_multiple_pcls(*[pc1[0].detach().cpu().numpy(), sync_pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy()])

instance_classes = torch.argmax(mask, dim=2).squeeze(0).detach().cpu().numpy()
visualize_points3D(pc1[0].detach().cpu().numpy(), instance_classes)
# visualize_points3D(pc1[0].detach().cpu().numpy(), instance_classes != 0)
# visualize_points3D(pc1[0].detach().cpu().numpy(), rmsd[0].detach().cpu().numpy())

