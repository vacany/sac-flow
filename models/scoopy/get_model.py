import argparse

from models.scoopy.networks.scoop import SCOOP

def build_scoop(args=None):

    # Args
    parser = argparse.ArgumentParser(description="Evaluate SCOOP.")
    parser.add_argument("--dataset_name", type=str, default="HPLFlowNet_kitti",
                        help="Dataset. FlowNet3D_kitti or FlowNet3D_FT3D or Either HPLFlowNet_kitti or HPLFlowNet_FT3D.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--mode", type=str, default="test",
                        help="Test or validation or all dataset (options: [val, test, all]).")
    parser.add_argument("--use_test_time_refinement", type=int, default=1,
                        help="1: Use test time refinement, 0: Do not use test time refinement.")
    parser.add_argument("--test_time_num_step", type=int, default=150, help="1: Number of steps for test time refinement.")
    parser.add_argument("--test_time_update_rate", type=float, default=0.2, help="1: Update rate for test time refinement.")
    parser.add_argument("--backward_dist_weight", type=float, default=0.0,
                        help="Backward distance weight for target reconstruction loss in test time refinement.")
    parser.add_argument("--target_recon_loss_weight", type=float, default=1.0,
                        help="Weight for target reconstruction loss in test time refinement.")
    parser.add_argument("--use_smooth_flow", type=int, default=1,
                        help="1: Use self smooth flow loss in test time refinement, 0: Do not use smooth flow loss.")
    parser.add_argument("--use_visibility_smooth_loss", type=int, default=0,
                        help="1: Use visibility-aware self smooth flow loss in test time refinement, 0: Do not use visibility smooth flow loss.")  # can be more specific to avoid nested args

    parser.add_argument("--nb_neigh_smooth_flow", type=int, default=32,
                        help="Number of neighbor points for smooth flow loss in test time refinement.")
    parser.add_argument("--smooth_flow_loss_weight", type=float, default=1.0,
                        help="Weight for smooth flow loss in test time refinement. Active if > 0.")
    parser.add_argument("--test_time_verbose", type=int, default=0,
                        help="1: Print test time results during optimization, 0: Do not print.")
    parser.add_argument("--use_chamfer_cuda", type=int, default=1,
                        help="1: Use chamfer distance cuda implementation in test time refinement, 0: Use chamfer distance pytorch implementation in test time refinement.")
    parser.add_argument("--nb_points", type=int, default=2048, help="Maximum number of points in point cloud.")
    parser.add_argument("--all_points", type=int, default=1,
                        help="1: use all point in the source point cloud for evaluation in chunks of nb_points, 0: use only nb_points.")
    parser.add_argument("--all_candidates", type=int, default=1,
                        help="1: use all points in the target point cloud as candidates concurrently, 0: use chunks of nb_points from the target point cloud each time.")
    parser.add_argument("--nb_points_chunk", type=int, default=2048,
                        help="Number of source points chuck for evaluation with all candidate target points.")
    parser.add_argument("--nb_workers", type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument("--exp_name", type=str, default='example',
                        help="Name of experiment, destination of experiment folder.")
    parser.add_argument("--path2ckpt", type=str, default=f"./../pretrained_models/kitti_v_100_examples/model_freespace.tar",
                        help="Path to saved checkpoint.")
    parser.add_argument("--log_fname", type=str, default="log_evaluation.txt", help="Evaluation log file name.")
    parser.add_argument("--save_pc_res", type=int, default=1,
                        help="1: save point cloud results, 0: do not save point cloud results [default: 0]")
    parser.add_argument("--res_folder", type=str, default="pc_res", help="Folder name for saving results.")
    parser.add_argument("--save_metrics", type=int, default=1,
                        help="1: save evaluation metrics results, 0: do not save evaluation metrics results [default: 0]")
    parser.add_argument("--metrics_fname", type=str, default="metrics_results.npz", help="Name for metrics file.")
    args = parser.parse_args()

    model = SCOOP(args)

    return model
