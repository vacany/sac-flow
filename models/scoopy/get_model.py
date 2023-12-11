import numpy as np
import torch
import os

from models.scoopy.networks.scoop import SCOOP
from models.scoopy.utils.utils import iterate_in_chunks
def build_scoop(dataset='kitti'):
    # Load Checkpoint
    if dataset == 'kitti':
        path_to_chkp = '/models/pretrained_models/scoop/kitti_v_100_examples/model_e400.tar'
    if dataset == 'ft3d':
        path_to_chkp = 'models/SCOOP/pretrained_models/ft3d_o_1800_examples/model_e100.tar'

    # path_to_chkp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/../../', path_to_chkp)
    path_to_chkp = '/home/vacekpa2/4D-RNSFP/' + path_to_chkp
    # print(path_to_chkp)
    file = torch.load(path_to_chkp, map_location='cpu')

    # Load parameters
    saved_args = file["args"]
    scoop = SCOOP(saved_args)
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # scoop = scoop.to(device, non_blocking=True)
    # scoop.device = device
    scoop.load_state_dict(file["model"])
    scoop = scoop.eval()

    return scoop

class PretrainedSCOOP(torch.nn.Module):

    def __init__(self):
        super(PretrainedSCOOP, self).__init__()
        self.scoop = build_scoop()


    def update(self, pc1, pc2):

        with torch.no_grad():

            # pc1 = pc1[0, :, ]
            # est_flow, corr_conf, graph = self.compute_flow(pc1, pc2) # do KITTISF now

            # k1, k2 = pc1[..., [1, 2, 0]], pc2[..., [1, 2, 0]]


            est_flow, corr_conf, graph = self.compute_flow(pc1[..., [1, 2, 0]], pc2[..., [1, 2, 0]])
            # est_flow, corr_conf, graph = self.compute_flow(k1, k2)

        # est_flow = est_flow[..., [1, 2, 0]] # switch back to original order
        est_flow = est_flow[..., [2, 0, 1]] # switch back to original order

        # from vis.deprecated_vis import visualize_flow3d
        # visualize_flow3d(pc1, pc2, est_flow)

        self.est_flow = est_flow.clone()
        self.refinement = torch.nn.Parameter(torch.zeros(pc1.shape, dtype=torch.float32, device=pc1.device, requires_grad=True))

        self.optimizer = torch.optim.Adam([self.refinement], lr=0.02)  # scoop lr is 0.02

    def forward(self, pc1, pc2=None, iters=150):
        # return self.est_flow

        return self.refinement + self.est_flow

    def compute_flow(self, pc_0, pc_1, nb_points_chunk=2048):
        # pc_0, pc_1 = batch["sequence"][0], batch["sequence"][1]

        n0 = int(pc_0.shape[1])
        n1 = int(pc_1.shape[1])


        with torch.no_grad():
            est_flow = torch.zeros([1, n0, 3], dtype=torch.float32, device=pc_0.device)
            corr_conf = torch.zeros([1, n0], dtype=torch.float32, device=pc_0.device)

            feats_0, graph = self.scoop.get_features(pc_0)
            feats_1, _ = self.scoop.get_features(pc_1)

            b, nb_points0, c = feats_0.shape
            b, nb_points1, c = feats_1.shape

            pc_0_orig = torch.unsqueeze(torch.reshape(pc_0, (b * nb_points0, 3))[:n0], dim=0)
            pc_1_orig = torch.unsqueeze(torch.reshape(pc_1, (b * nb_points1, 3))[:n1], dim=0)
            feats_0_orig = torch.unsqueeze(torch.reshape(feats_0, (b * nb_points0, c))[:n0], dim=0)
            feats_1_orig = torch.unsqueeze(torch.reshape(feats_1, (b * nb_points1, c))[:n1], dim=0)
            idx = np.arange(n0)
            for b in iterate_in_chunks(idx, nb_points_chunk):

                points = pc_0_orig[:, b]
                feats = feats_0_orig[:, b]
                points_flow, points_conf, _ = self.scoop.get_recon_flow([points, pc_1_orig], [feats, feats_1_orig])
                est_flow[:, b] = points_flow
                corr_conf[:, b] = points_conf

        return est_flow, corr_conf, graph


