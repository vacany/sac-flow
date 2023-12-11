import itertools
import pandas as pd

permutations1a = {'dataset_type': ['kitti_t', 'kitti_o'],
                    'model' : ['NP'],
                    'lr': [0.001],
                    'K': [32],                                        
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [1],
                    'forward_weight': [10],
                    'sm_normals_K': [4],                    
                    'early_patience': [0],
                    'max_iters': [400],
                    'runs': [1],
                    
                    }

permutations1b = {'dataset_type': ['kitti_t', 'kitti_o'],
                    'model' : ['SCOOP'],
                    'lr': [0.2],
                    'K': [32],
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [10],
                    'forward_weight': [10],
                    'sm_normals_K': [4],
                    'early_patience': [0],
                    'max_iters': [400],
                    'runs': [1],

                    }

permutations2 = {'dataset_type': ['argoverse', 'nuscenes', 'waymo'],
                    'model' : ['NP'],
                    'lr': [0.008],
                    'K': [4],    
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [1],
                    'forward_weight': [10],
                    'sm_normals_K': [4],                
                    'early_patience': [0],
                    'max_iters': [400],
                    'runs': [1],
                    
                    }

def generate_configs(permutations):
    ''' Create configs for parameter grid search, or dataset cross evaluation 
        Function returs pandas dataframe, where each row is one experiment config
        This structure can be easily exploited on SLURM job array sbatch'''
    
    combinations1 = list(itertools.product(*permutations.values()))


    df = pd.DataFrame(combinations1, columns=permutations.keys())

    return df

cfg1a = generate_configs(permutations=permutations1a)
cfg1b = generate_configs(permutations=permutations1b)
cfg2 = generate_configs(permutations=permutations2)

cfg = pd.concat((cfg1a, cfg1b, cfg2))