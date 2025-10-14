# MMSegmentation Training Pipeline

A comprehensive training and inference pipeline for semantic segmentation using MMSegmentation framework. This repository supports multiple state-of-the-art models for sidewalk, crosswalk, and road segmentation tasks.

## Table of Contents
- [Supported Models](#supported-models)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset Structure](#dataset-structure)
- [Running Training](#running-training)
- [Inference Only](#inference-only)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

## Supported Models

The pipeline supports the following pre-trained models:

**DeepLabV3+**
- `deeplabv3plus_r50` - ResNet-50 backbone
- `deeplabv3plus_r101` - ResNet-101 backbone

**PSPNet**
- `pspnet_r50` - ResNet-50 backbone
- `pspnet_r101` - ResNet-101 backbone

**HRNet**
- `hrnet_w18` - HRNet-W18-Small
- `hrnet_w48` - HRNet-W48

**OCRNet**
- `ocrnet` - HRNet-W48 backbone

**SegFormer**
- `segformer_b0` - MiT-B0 backbone
- `segformer_b5` - MiT-B5 backbone

**Swin Transformer**
- `swin_t` - Swin-Tiny
- `swin_b` - Swin-Base

**SETR**
- `setr_mla` - Vision Transformer with Multi-Level Aggregation
- `setr_pup` - Vision Transformer with Progressive UPsampling

**Mask2Former**
- `mask2former_r101` - ResNet-101 backbone
- `mask2former_swins` - Swin-Small backbone

## Prerequisites

### Required Software
- Singularity/Apptainer (for containerized execution)
- SLURM (for job scheduling)
- CUDA-capable GPU

### Singularity Container
You need the MMSegmentation Singularity container(TBA). Update the path in your scripts:
```bash
CONTAINER=/home/pss442/mmseg.sif
```

### Important: Workspace Path Binding
⚠️ **Critical Configuration**: The `WORKSPACE` variable in your shell scripts must match the bind mount path.

In your `.sh` scripts, ensure consistency:
```bash
WORKSPACE=/scratch/$USER/Sidewalk_Dataset

# This binds your local workspace to /workspace inside the container
singularity exec --nv \
    --containall \
    --bind "$WORKSPACE:/workspace" \
    ...
```

**Key Points:**
- `WORKSPACE` is your **local** path (outside container), which should be the cloned sidewalk dataset directory
- `/workspace` is the **container** path (inside container)
- All paths in Python script arguments use `/workspace` (e.g., `DATA_ROOT="/workspace/data"`)
- If you change `WORKSPACE`, update it consistently in all `.sh` scripts
- If you change the bind target (`:workspace`), update all paths in the scripts accordingly

## Setup

### 1. Clone the Repository
```bash
cd /scratch/$USER
git clone <your-repo-url> Sidewalk_Dataset
cd Sidewalk_Dataset
```

### 2. Clone MMSegmentation
Clone the MMSegmentation repository into your workspace:
```bash
cd /scratch/$USER/Sidewalk_Dataset
git clone https://github.com/open-mmlab/mmsegmentation.git
```

**Important**: The training script expects MMSegmentation configs at:
```
/workspace/mmsegmentation/configs/
```

### 3. Create Required Directories
```bash
# Create logs directory
mkdir -p logs

# Create torch cache directory (for model checkpoints)
mkdir -p torch_cache

# Create data directory structure (if not already present)
mkdir -p data/train/images data/train/masks/gray
mkdir -p data/val/images data/val/masks/gray
mkdir -p data/test/images data/test/masks/gray

# Create results directory
mkdir -p results
```

### 4. Verify Directory Structure
Your workspace should look like this:
```
Sidewalk_Dataset/
├── train_and_inference.py
├── run_training.sh
├── test_install.sh
├── train_array.sh
├── logs/
├── torch_cache/
├── mmsegmentation/
│   └── configs/
│       ├── deeplabv3plus/
│       ├── pspnet/
│       ├── hrnet/
│       └── ...
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/gray/
│   ├── val/
│   │   ├── images/
│   │   └── masks/gray/
│   └── test/
│       ├── images/
│       └── masks/gray/
└── results/
```

## Dataset Structure

### Data Organization
Place your images and masks in the following structure:

```
data/
├── train/
│   ├── images/           # Training images (.png, .jpg, etc.)
│   └── masks/
│       └── gray/         # Grayscale masks (.png)
├── val/
│   ├── images/           # Validation images
│   └── masks/
│       └── gray/         # Grayscale masks
└── test/
    ├── images/           # Test images
    └── masks/
        └── gray/         # Grayscale masks (optional)
```

### Mask Format
- Grayscale images with pixel values representing class IDs
- Class mapping:
  - `0`: Background
  - `1`: Sidewalk
  - `2`: Road
  - `3`: Crosswalk

## Running Training

### Single Model Training

Edit `run_training.sh` to configure your model and parameters:
```bash
MODEL="deeplabv3plus_r50"  # Choose from supported models
BATCH_SIZE=8
MAX_ITERS=40000
VAL_INTERVAL=1000
IMG_SIZE=512
```

Submit the job:
```bash
sbatch run_training.sh
```

### Array Job Training (Multiple Models)

To train multiple models in parallel, use the array job script:
```bash
sbatch train_array.sh
```

Edit `train_array.sh` to specify which models to train:
```bash
MODELS=("deeplabv3plus_r50" "hrnet_w48" "segformer_b5")
```

### Monitor Training
```bash
# View output logs
tail -f logs/mmseg_cvpr_dataset_train_<job_id>.out

# View error logs
tail -f logs/mmseg_cvpr_dataset_train_<job_id>.err

# Check job status
squeue -u $USER
```

## Configuration

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | Required |
| `--data_root` | Root directory of dataset | Required |
| `--results_dir` | Output directory | `results` |
| `--batch_size` | Training batch size | `8` |
| `--num_workers` | Data loading workers | `4` |
| `--max_iters` | Maximum training iterations | `40000` |
| `--val_interval` | Validation frequency | `1000` |
| `--lr` | Learning rate (0 = use default) | `0` |
| `--img_size` | Input image size | `512` |
| `--num_classes` | Number of classes | `4` |

### Pretrained Weights

By default, models use pretrained weights from the model zoo:
```bash
# Use pretrained weights (default)
--use_pretrained

# Train from scratch
--no_pretrained

# Use custom pretrained weights
--pretrained /path/to/weights.pth
```

### Resume Training
```bash
# Add to your training command
--resume
```

## Inference Only

### Run Inference with Trained Model
```bash
singularity exec --nv \
    --containall \
    --bind "$WORKSPACE:/workspace" \
    $CONTAINER python /workspace/train_and_inference.py \
    --model deeplabv3plus_r50 \
    --data_root /workspace/data \
    --results_dir /workspace/results \
    --inference_only \
    --checkpoint /workspace/results/model_timestamp/best_mIoU_iter_*.pth \
    --img_size 512 \
    --num_classes 4
```

### Generate Predictions During Training
Add the `--save_predictions` flag to save predictions after training:
```bash
--save_predictions
```

## Output Structure

After training, results are organized as follows:

```
results/
└── <model_name>_<timestamp>/
    ├── config.py              # Full training configuration
    ├── args.json              # Training arguments
    ├── metrics.json           # Validation and test metrics
    ├── best_mIoU_iter_*.pth  # Best model checkpoint
    ├── latest.pth             # Latest checkpoint
    ├── tf_logs/               # TensorBoard logs
    ├── vis_data/              # Visualization data
    ├── predictions/           # Predicted masks (if --save_predictions)
    │   ├── test/
    │   └── val/
    └── visualizations/        # Side-by-side visualizations
        ├── test/
        └── val/
```

### Visualizations
Each visualization shows:
- Original image | Ground truth mask | Predicted mask

Colors:
- Black: Background
- Blue: Sidewalk
- Green: Road
- Red: Crosswalk

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Ensure PYTHONPATH is cleared in your script
unset PYTHONPATH
unset PYTHONHOME
unset PYTHON_PATH
```

**2. Config file not found**
- Verify MMSegmentation is cloned in workspace
- Check path: `/workspace/mmsegmentation/configs/`

**3. Out of memory errors**
- Reduce `BATCH_SIZE`
- Reduce `IMG_SIZE`
- Use a smaller model variant

**4. Checkpoint not loading**
- Verify checkpoint path exists
- Check that `num_classes` matches trained model

**5. CUDA out of memory**
```bash
# Reduce batch size in run_training.sh
BATCH_SIZE=4  # or lower
```

### Testing Installation
```bash
sbatch test_install.sh
```

This will verify:
- Singularity container accessibility
- Python environment
- MMSegmentation installation
- CUDA availability

### Check Logs
```bash
# List all job logs
ls -lh logs/

# View specific job output
cat logs/mmseg_cvpr_dataset_train_<job_id>.out
```

## Performance Tips

1. **Batch Size**: Start with 8 and adjust based on GPU memory
2. **Learning Rate**: Use 0 to keep model defaults, or tune carefully
3. **Image Size**: 512x512 is a good balance of quality and speed
4. **Validation Interval**: Check every 500-1000 iterations
5. **Multi-GPU**: Modify batch size accordingly when using multiple GPUs

## Citation

If you use this code, please cite MMSegmentation:
```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project follows the same license as MMSegmentation (Apache 2.0).

## Support

For issues specific to:
- **This pipeline**: Open an issue in this repository
- **MMSegmentation**: Refer to [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/)
- **SLURM/HPC**: Contact your cluster administrators