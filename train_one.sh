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
#SBATCH --account=YOUR ACCOUNT HERE

#update the account above (if needed by your slurm settings)



echo "=================================================="
echo "Starting training for model: $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Paths
#downlaod the container and update the container path here
CONTAINER=/home/pss442/mmseg.sif

#set your workspace to the cloned repo
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
MAX_ITERS=2000
VAL_INTERVAL=250
LR=0
IMG_SIZE=512
NUM_CLASSES=4
MODEL="deeplabv3plus_r101"  # Options: deeplabv3plus_r50, deeplabv3plus_r101, hrnet_w18, hrnet_w48, pspnet_r50, pspnet_r101, ocrnet, segformer_b0, segformer_b5, swin_b, swin_t, setr_pup, setr_mla, mask2former_r101, mask2former_swins

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data root: $DATA_ROOT"
echo "  Results dir: $RESULTS_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Max iterations: $MAX_ITERS"
echo "  Image size: $IMG_SIZE"
echo ""

# Setup SETR pretrained weights if needed
if [[ "$MODEL" == "setr_pup" ]] || [[ "$MODEL" == "setr_mla" ]]; then
    echo "Setting up SETR pretrained weights..."
    
    # Create pretrain directory
    mkdir -p $WORKSPACE/pretrain
    
    # Check if converted checkpoint exists
    if [ ! -f "$WORKSPACE/pretrain/vit_large_p16.pth" ]; then
        echo "Downloading and converting ViT checkpoint..."
        
        # Download original checkpoint
        if [ ! -f "$WORKSPACE/pretrain/jx_vit_large_p16_384-b3be5167.pth" ]; then
            wget -P $WORKSPACE/pretrain https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth
        fi
        
        # Convert downloaded VIT weights to mmseg checkpoint
        singularity exec --nv \
            --containall \
            --bind "$WORKSPACE:/workspace" \
            $CONTAINER python /workspace/mmsegmentation/tools/model_converters/vit2mmseg.py \
            /workspace/pretrain/jx_vit_large_p16_384-b3be5167.pth \
            /workspace/pretrain/vit_large_p16.pth
    else
        echo "Converted ViT checkpoint already exists, skipping download."
    fi
fi

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