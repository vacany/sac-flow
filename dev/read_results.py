import os
import glob
import pandas as pd

def load_results_to_df(exp_names : list):

    runs = []

    for exp_name in exp_names:
        exp_dir = f'{os.path.expanduser("~")}/experiments/{exp_name}'
        runs += (sorted(glob.glob(f'{exp_dir}/*')))

    # exp_name = "lidar_grid_search"
    # exp_dir = f'{os.path.expanduser("~")}/experiments/{exp_name}'
    # runs.append(sorted(glob.glob(f'{exp_dir}/*')))

    metric_all = []

    for run in runs:

        if os.path.exists(f'{run}/metric.csv') == False:
            continue

        metric = pd.read_csv(f'{run}/metric.csv', index_col=0, header=None).transpose()
        metric = [v.to_dict() for k,v in metric.iterrows()][0]

        args = pd.read_csv(f'{run}/args.csv', header=None, index_col=0).transpose()
        args = [v.to_dict() for k, v in args.iterrows()][0]

        for k in ['VA', 'free_weight', 'pc2_smooth', 'smooth_weight', 'forward_weight', 'sm_normals_K']:
            if k in args.keys():
                args[k] = float(args[k])

        # Skip tryouts
        if args['dev'] == 1:
            continue

        epe = metric['EPE3D']
        avg_solver_time = metric['avg_solver_time']
        # decide about aff
        if args['model'] == 'NeuralPrior':

            if args['VA'] == 0 and args['free_weight'] == 0 and args['forward_weight'] == 0 and args['pc2_smooth'] == 0 and args['smooth_weight'] == 0:
                aff = 'baseline'
            else:
                aff = 'ours'

        if args['model'] == 'SCOOP':
            if args['sm_normals_K'] == 0 and args['VA'] == 0 and args['free_weight'] == 0 and args['forward_weight'] == 0 and args['pc2_smooth'] == 0:
                aff = 'baseline'
            else:
                aff = 'ours'

        # if aff == 'baseline':
        #     continue

        del args['affiliation']
        args['aff'] = aff


        full_dict = {**args, **metric}
        name_weight_list = ['dataset', 'free_weight', 'smooth_weight', 'forward_weight', 'pc2_smooth', 'sm_normals_K', 'VA']
        name_metrics = ['EPE3D', 'acc3d_strict', 'acc3d_relax', 'angle_error', 'outlier']
        interest_dict = {k:v for k,v in full_dict.items() if k in ['aff', 'avg_solver_time', 'model', 'EPE3D', 'iters'] + name_weight_list + name_metrics}

        # metric_all.append(full_dict)
        metric_all.append(interest_dict)

    df = pd.DataFrame(metric_all)

    return df



# exp_names = ['lidar_grid_search', 'grid_search']
# exp_names = ['kitti_o_FT']
# df = load_results_to_df(exp_names)
