#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=8         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --partition=amdgpufast      # partition name
#SBATCH --gres=gpu:1                 # 1 gpu per node
#SBATCH --mem=20G
#SBATCH --error=log/myJob.err            # standard error file
#SBATCH --output=log/myJob.out           # standard output file
#SBATCH --array 0-3%20

ml PyTorch3D/0.7.3-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0

cd $HOME/pcflow/

python scripts/single_gpu.py ${SLURM_ARRAY_TASK_ID}
