import os
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS
import torch

# Register all modules
register_all_modules()

# Model zoo with base configs and pretrained checkpoints
MODEL_CONFIGS = {

    'deeplabv3plus_r50': {
    'base': 'deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_potsdam-512x512.py',
    'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_potsdam/deeplabv3plus_r50-d8_512x512_80k_potsdam_20211219_031508-7e7a2b24.pth'    
    },

    'deeplabv3plus_r101': {
        'base': 'deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_potsdam/deeplabv3plus_r101-d8_512x512_80k_potsdam_20211219_031508-8b112708.pth'
    },

    'pspnet_r50': {
        'base': 'pspnet/pspnet_r50-d8_4xb4-80k_potsdam-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_4x4_512x512_80k_potsdam/pspnet_r50-d8_4x4_512x512_80k_potsdam_20211219_043541-2dd5fe67.pth'
    },
    'pspnet_r101': {
        'base': 'pspnet/pspnet_r101-d8_4xb4-80k_potsdam-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_4x4_512x512_80k_potsdam/pspnet_r101-d8_4x4_512x512_80k_potsdam_20211220_125612-aed036c4.pth'   
    },

    'hrnet_w18': {
        'base': 'hrnet/fcn_hr18s_4xb4-80k_potsdam-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_80k_potsdam/fcn_hr18s_512x512_80k_potsdam_20211218_205517-ba32af63.pth'
    },
    'hrnet_w48': {
        'base': 'hrnet/fcn_hr48_4xb4-80k_potsdam-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_80k_potsdam/fcn_hr48_512x512_80k_potsdam_20211219_020601-97434c78.pth'
    },

    'ocrnet': {
        'base': 'ocrnet/ocrnet_hr48_4xb4-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x512_160k_ade20k/ocrnet_hr48_512x512_160k_ade20k_20200615_184705-a073726d.pth'
    },

    'segformer_b0': {
        'base': 'segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth'
    },
    'segformer_b5': {
        'base': 'segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'
    },
    
    'swin_t': {
        'base': 'swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
    },
    'swin_b': {
        'base': 'swin/swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth'
    },

    'setr_mla': {
        'base': 'setr/setr_vit-l-mla_8xb1-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth'
    },
    'setr_pup': {
        'base': 'setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth'
    },

    'mask2former_r101': {
        'base': 'mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth'
    },
    'mask2former_swins': {
        'base': 'mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py',
        'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth'
    },
    
}


@DATASETS.register_module()
class SidewalkDataset(BaseSegDataset):
    """Custom dataset for sidewalk, crosswalk, and road segmentation."""
    
    METAINFO = {
        'classes': ('background', 'sidewalk', 'road', 'crosswalk'),
        'palette': [[0, 0, 0], [0, 0, 255], [0, 128, 0], [255, 0, 0]]
    }
    
    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )

def create_config(args):
    """
    Create MMSegmentation config

    """
    
    # Validate model
    if args.model not in MODEL_CONFIGS:
        raise ValueError(
            f"Model '{args.model}' not supported. "
            f"Available models: {', '.join(MODEL_CONFIGS.keys())}"
        )
    
    # Load base config from mmseg
    base_cfg_path = f"/workspace/mmsegmentation/configs/{MODEL_CONFIGS[args.model]['base']}"
    cfg = Config.fromfile(base_cfg_path)
        
    # Update number of classes for decode heads
    # 
    if isinstance(cfg.model.decode_head, list):
        for decode_head in cfg.model.decode_head:
            decode_head.num_classes = args.num_classes
    else:
        cfg.model.decode_head.num_classes = args.num_classes

    # Update number of classes for auxiliary heads
    if hasattr(cfg.model, 'auxiliary_head') and cfg.model.auxiliary_head is not None:
        # Handle both single auxiliary head and list of auxiliary heads
        if isinstance(cfg.model.auxiliary_head, list):
            for aux_head in cfg.model.auxiliary_head:
                aux_head.num_classes = args.num_classes
        else:
            cfg.model.auxiliary_head.num_classes = args.num_classes
    
    # Load pretrained checkpoint
    if getattr(args, 'use_pretrained', True):
        cfg.load_from = MODEL_CONFIGS[args.model]['checkpoint']
        
        # For SETR models, also set the backbone checkpoint path
        if args.model in ['setr_pup', 'setr_mla']:
            if hasattr(cfg.model, 'backbone'):
                # Update or create backbone init_cfg
                if hasattr(cfg.model.backbone, 'init_cfg') and cfg.model.backbone.init_cfg is not None:
                    if isinstance(cfg.model.backbone.init_cfg, dict):
                        cfg.model.backbone.init_cfg['checkpoint'] = '/workspace/pretrain/vit_large_p16.pth'
                    else:
                        cfg.model.backbone.init_cfg = dict(
                            type='Pretrained',
                            checkpoint='/workspace/pretrain/vit_large_p16.pth'
                        )
                else:
                    cfg.model.backbone.init_cfg = dict(
                        type='Pretrained',
                        checkpoint='/workspace/pretrain/vit_large_p16.pth'
                    )
    
    # Update dataset paths
    cfg.train_dataloader.dataset.type = 'SidewalkDataset'
    cfg.train_dataloader.dataset.data_root = args.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(
        img_path='train/images',
        seg_map_path='train/masks/gray'
    )
    cfg.train_dataloader.dataset.reduce_zero_label = False  

    cfg.val_dataloader.dataset.type = 'SidewalkDataset'
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path='val/images',
        seg_map_path='val/masks/gray'
    )
    cfg.val_dataloader.dataset.reduce_zero_label = False  

    cfg.test_dataloader.dataset.type = 'SidewalkDataset'

    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_prefix = dict(
        img_path='test/images',
        seg_map_path='test/masks/gray'
    )
    cfg.test_dataloader.dataset.reduce_zero_label = False  

    #Set val and test batch size
    cfg.val_dataloader.batch_size = 128
    cfg.test_dataloader.batch_size = 128

    # ========== ARG OVERRIDES ==========
    
    # Batch size
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    
    
    # Learning rate 
    if hasattr(args, 'lr') and args.lr is not None and args.lr > 0:
        cfg.optim_wrapper.optimizer.lr = args.lr
    
    # Max iterations
    if hasattr(args, 'max_iters') and args.max_iters is not None:
        cfg.train_cfg.max_iters = args.max_iters
        # Update scheduler end point
        if hasattr(cfg, 'param_scheduler'):
            for scheduler in cfg.param_scheduler:
                if 'end' in scheduler:
                    scheduler['end'] = args.max_iters
    
    # Validation interval
    if hasattr(args, 'val_interval') and args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval
        cfg.default_hooks.checkpoint.interval = args.val_interval
    
    # Image size 
    if hasattr(args, 'img_size') and args.img_size is not None:
        # Update crop size in data preprocessor
        if hasattr(cfg.model, 'data_preprocessor'):
            cfg.model.data_preprocessor.size = (args.img_size, args.img_size)
        
        # Update pipelines
        for pipeline_cfg in cfg.train_dataloader.dataset.pipeline:
            if pipeline_cfg['type'] == 'RandomCrop':
                pipeline_cfg['crop_size'] = (args.img_size, args.img_size)
            elif pipeline_cfg['type'] == 'RandomResize':
                pipeline_cfg['scale'] = (args.img_size * 2, args.img_size * 2)
            if pipeline_cfg['type'] == 'LoadAnnotations': 
                pipeline_cfg['reduce_zero_label'] = False
        
        for pipeline_cfg in cfg.val_dataloader.dataset.pipeline:
            if pipeline_cfg['type'] == 'Resize':
                pipeline_cfg['keep_ratio'] = False
                pipeline_cfg['scale'] = (args.img_size, args.img_size)
            if pipeline_cfg['type'] == 'LoadAnnotations': 
                pipeline_cfg['reduce_zero_label'] = False


        for pipeline_cfg in cfg.test_dataloader.dataset.pipeline:
            if pipeline_cfg['type'] == 'LoadAnnotations': 
                pipeline_cfg['reduce_zero_label'] = False

        # Remove Resize from test pipeline -
        cfg.test_dataloader.dataset.pipeline = [
            p for p in cfg.test_dataloader.dataset.pipeline 
            if p['type'] != 'Resize'
        ]

    # Work directory 
    if hasattr(args, 'work_dir') and args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # Resume training (if specified)
    if hasattr(args, 'resume') and args.resume:
        cfg.resume = args.resume

    # Checkpoint Settings
    if hasattr(cfg, "default_hooks") and "checkpoint" in cfg.default_hooks:
        cfg.default_hooks.checkpoint.update(
            dict(save_best='mIoU', rule='greater')
        )
    else:
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook',
            interval=1,
            by_epoch=False,
            save_best='mIoU',
            rule='greater'
        )
    
    return cfg


def run_inference(model, data_root, split, output_dir, img_size):
    """Run inference and save predictions."""
    img_dir = Path(data_root) / split / 'images'
    mask_dir = Path(data_root) / split / 'masks/gray'
    pred_dir = Path(output_dir) / 'predictions' / split
    vis_dir = Path(output_dir) / 'visualizations' / split
    
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(img_dir.glob('*'))
    
    print(f"\nRunning inference on {split} set ({len(img_files)} images)...")
    
    for img_file in img_files:
        # Run inference
        result = inference_model(model, str(img_file))
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        
        # Save prediction
        pred_file = pred_dir / img_file.name
        cv2.imwrite(str(pred_file), pred_mask.astype(np.uint8))
        
        # Create visualization
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load ground truth if available
        mask_file = mask_dir / img_file.name
        if mask_file.exists():
            gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = None
                
        # Create color overlay
        colors = np.array([[0, 0, 0], [0, 0, 255], [0, 128, 0], [255, 0, 0]], dtype=np.uint8)
        pred_colored = colors[pred_mask]

        # Create side-by-side visualization
        if gt_mask is not None:
            gt_colored = colors[gt_mask]
            combined = np.hstack([img, gt_colored, pred_colored])
        else:
            combined = np.hstack([img, pred_colored])

        vis_file = vis_dir / img_file.name
        cv2.imwrite(str(vis_file), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            
    print(f"Inference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeplabv3plus_r50', 'deeplabv3plus_r101', 'hrnet_w18', 'hrnet_w48' , 'pspnet_r50', 'pspnet_r101', 'ocrnet', 'segformer_b0', 'segformer_b5', 'swin_b', 'swin_t', 'setr_pup', 'setr_mla', 'mask2former_r101', 'mask2former_swins' ],
                        help='Model architecture to use')
    
    # Data paths
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_iters', type=int, default=40000,
                        help='Maximum training iterations')
    parser.add_argument('--val_interval', type=int, default=1000,
                        help='Validation interval')
    parser.add_argument('--lr', type=float, default=0,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes (including background)')
    
    # Checkpoint and resume
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (overrides default pretrained weights)')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='Use pretrained weights from model zoo (default: True)')
    parser.add_argument('--no_pretrained', dest='use_pretrained', action='store_false',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    # Inference
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions on test set')
    parser.add_argument('--inference_only', action='store_true',
                        help='Only run inference without training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for inference')
    
    args = parser.parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{timestamp}"
    work_dir = Path(args.results_dir) / exp_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(work_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    if not args.inference_only:
        # Create config
        cfg = create_config(args)
        cfg.work_dir = str(work_dir)
        
        # Save config
        cfg.dump(str(work_dir / 'config.py'))
        
        print(f"Starting training {args.model}...")
        print(f"Results will be saved to: {work_dir}")
        
        # Build runner and start training
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("\nTraining completed!")
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = runner.val()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = runner.test()
        
        # Save metrics
        metrics = {
            'val': val_metrics,
            'test': test_metrics
        }
        
        with open(work_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nValidation mIoU: {val_metrics.get('mIoU', 'N/A')}")
        print(f"Test mIoU: {test_metrics.get('mIoU', 'N/A')}")
        
        best_checkpoint = work_dir / 'best_mIoU_iter_*.pth'
        checkpoint_path = list(work_dir.glob('best_mIoU_iter_*.pth'))
        
        if checkpoint_path and args.save_predictions:
            checkpoint_path = str(checkpoint_path[0])
    else:
        checkpoint_path = args.checkpoint
    
    # Run inference if requested
    if args.save_predictions or args.inference_only:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            cfg = create_config(args)
            model = init_model(cfg, checkpoint_path, device='cuda:0')
            
            # Run inference on test set
            run_inference(model, args.data_root, 'test', work_dir, args.img_size)
            
            # Optionally run on validation set too
            if args.inference_only:
                run_inference(model, args.data_root, 'val', work_dir, args.img_size)
        else:
            print("Warning: No checkpoint found for inference!")
    
    print(f"\n{'='*50}")
    print(f"All tasks completed!")
    print(f"Results saved to: {work_dir}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()