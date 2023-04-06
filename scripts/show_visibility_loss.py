import numpy as np
import os
from ops.visibility import mask_KNN_by_visibility



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cfg = {"x_min" : -50,
           "x_max" : 50,
           "y_min" : -50,
           "y_max" : 50,
           "z_min" : -5,
           "z_max" : 5,
           "cell_size" : (0.15,0.15,0.15),
           "size_of_block" : 0,
           }

    frame = 162  # 19 ICP failed
    data = np.load(f'/home/patrik/SCOOP/pretrained_models/kitti_v_100_examples/pc_res/{frame:06d}_res..npz', allow_pickle=True)

    pc1 = data['pc1'][:, [0, 2, 1]]
    pc2 = data['pc2'][:, [0, 2, 1]]
    gt_mask1 = data['gt_mask_for_pc1']
    gt_flow1 = data['gt_flow_for_pc1'][:, [0, 2, 1]]
    est_flow1 = data['est_flow_for_pc1'][:, [0, 2, 1]]
    corr_conf1 = data['corr_conf_for_pc1']


    pts = pc2.copy()
    margin = cfg['cell_size'][0] * 2

    K = 32

    masked_KNN = mask_KNN_by_visibility(pc2, K, cfg, margin=margin)
    from sklearn.neighbors import NearestNeighbors
    orig_KNN = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(pts).kneighbors(pts[:, :3], return_distance=True)

    np.savez('/home/patrik/pcflow/models/SCOOP/scripts/KNN.npz', masked_KNN=masked_KNN, orig_KNN=orig_KNN[1])
