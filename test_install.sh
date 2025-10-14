#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name=mmseg_gpu
#SBATCH --output=mmseg_gpu.out
#SBATCH --account=pr_236_tandon_priority

# Paths
CONTAINER=/home/pss442/mmseg.sif
WORKSPACE=/scratch/$USER/Sidewalk_Dataset
unset PYTHONPATH
unset PYTHONHOME
unset PYTHON_PATH

singularity exec --nv --containall \
    --bind "$WORKSPACE:/workspace" \
    $CONTAINER python -c "
import mmseg
print(mmseg.__version__)
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import pandas as pd
print(f'Pandas: {pd.__version__}')
import tabulate
print('Tabulate: installed successfully')
"
