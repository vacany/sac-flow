# default params by models
# NeuralPrior
NeuralPrior_cfg = {'dev' : 0,
                   'exp_name' : 'NeuralPrior',
                   'pc2_smooth' : 0,
                   'iters' : 5000,
                   'model' : 'NeuralPrior',
                   'lr' : 0.008,
                   'smooth_weight' : 0,
                   'K' : 0,
                   'per_sample_init' : 1,
                   'both_ways' : 1,
                   'early_patience' : 30,
                   'early_min_delta' : 0.001,
                   'forward_weight' : 0}
# SCOOP
SCOOP_cfg = {'dev' : 0,
             'exp_name' : 'SCOOP',
             'pc2_smooth' : 0,
             'iters' : 150,
             'model' : 'SCOOP',
             'lr' : 0.2,
             'smooth_weight' : 10,
             'K' : 32,
             'max_radius' : 35, # This is for KNN only, not point cloud itself
             'per_sample_init' : 0,
             'both_ways' : 0,
             'early_patience' : 150,
             'early_min_delta' : 0.001,
             'forward_weight' : 0}
