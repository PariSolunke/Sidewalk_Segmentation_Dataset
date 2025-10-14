#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name=cvpr_dataset_train
#SBATCH --output=logs/mmseg_%x_%j.out
#SBATCH --error=logs/mmseg_%x_%j.err
#SBATCH --account=YOUR_ACCOUNT_HERE





echo "=================================================="
echo "Starting training for model: $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Paths

CONTAINER=/home/pss442/mmseg.sif
WORKSPACE=/scratch/$USER/Sidewalk_Dataset

# Clear Python environment variables
unset PYTHONPATH
unset PYTHONHOME
unset PYTHON_PATH

# Create logs directory if it doesn't exist
mkdir -p logs

# Training parameters (modify as needed)
DATA_ROOT="/workspace/data"
RESULTS_DIR="/workspace/results"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_ITERS=4000
VAL_INTERVAL=500
LR=0
IMG_SIZE=512
NUM_CLASSES=4
MODEL="deeplabv3plus_r101" 

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data root: $DATA_ROOT"
echo "  Results dir: $RESULTS_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Max iterations: $MAX_ITERS"
echo "  Image size: $IMG_SIZE"
echo ""

# Run training
singularity exec --nv \
    --containall \
    --bind "$WORKSPACE:/workspace" \
    --env TORCH_HOME=/workspace/torch_cache \
    $CONTAINER python /workspace/train_and_inference.py \
    --model $MODEL \
    --data_root $DATA_ROOT \
    --results_dir $RESULTS_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_iters $MAX_ITERS \
    --val_interval $VAL_INTERVAL \
    --lr $LR \
    --img_size $IMG_SIZE \
    --num_classes $NUM_CLASSES \
    --save_predictions
    
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully for $MODEL"
    echo "=================================================="
else
    echo "=================================================="
    echo "Training failed for $MODEL with exit code $EXIT_CODE"
    echo "=================================================="
fi

exit $EXIT_CODE