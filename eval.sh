#!/bin/bash
set -e

TAG="P4_P0"

# Run Panoptic evaluation
python train_net.py --eval-only --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_lars_coco.yaml --num-gpus 8 --tag "$TAG"

# Extract per class metrics
python extract_per_class_metrics.py --tag "$TAG"