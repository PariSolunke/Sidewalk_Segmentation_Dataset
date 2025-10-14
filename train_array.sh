#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name=cvpr_array_train
#SBATCH --output=logs/mmseg_array_%A_%a.out
#SBATCH --error=logs/mmseg_array_%A_%a.err
#SBATCH --account=pr_236_tandon_priority
#SBATCH --array=0-2%3

# Array job for training multiple models in parallel
# Usage: sbatch train_array.sh
# The %3 means max 3 jobs running simultaneously

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Define array of models to train
MODELS=(
    "deeplabv3plus_r50"
    "deeplabv3plus_r101"
    "pspnet_r50"
    "pspnet_r101"
    "hrnet_w18"
    "hrnet_w48"
    "ocrnet"
    "segformer_b0"
    "segformer_b5"
    "swin_t"
    "swin_b"
    "setr_mla"
    "setr_pup"
    "mask2former_r101"
    "mask2former_swins"
)

# Select model based on array task ID
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Training model: $MODEL"
echo ""

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
MAX_ITERS=40000
VAL_INTERVAL=1000
LR=0
IMG_SIZE=512
NUM_CLASSES=4

# Optional: Adjust batch size for specific models that need more memory
case $MODEL in
    "mask2former_r101"|"mask2former_swins"|"swin_b"|"setr_mla"|"setr_pup")
        BATCH_SIZE=4
        echo "Using reduced batch size ($BATCH_SIZE) for memory-intensive model"
        ;;
    "segformer_b5")
        BATCH_SIZE=6
        echo "Using reduced batch size ($BATCH_SIZE) for memory-intensive model"
        ;;
esac

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data root: $DATA_ROOT"
echo "  Results dir: $RESULTS_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Max iterations: $MAX_ITERS"
echo "  Validation interval: $VAL_INTERVAL"
echo "  Learning rate: $LR (0 = use default)"
echo "  Image size: $IMG_SIZE"
echo "  Number of classes: $NUM_CLASSES"
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
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "=================================================="
else
    echo "=================================================="
    echo "Training failed for $MODEL with exit code $EXIT_CODE"
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "=================================================="
fi

exit $EXIT_CODE