import sys
import subprocess
import wandb


def build_config_overrides(config):
    """Return list of config overrides for FC-CLIP training."""
    overrides = []
    
    # Learning rate parameters
    if 'base_lr' in config:
        overrides.extend(["SOLVER.BASE_LR", str(float(config.base_lr))])
    
    if 'weight_decay' in config:
        overrides.extend(["SOLVER.WEIGHT_DECAY", str(float(config.weight_decay))])
    
    if 'backbone_multiplier' in config:
        overrides.extend(["SOLVER.BACKBONE_MULTIPLIER", str(float(config.backbone_multiplier))])
    
    # Training parameters
    if 'max_iter' in config:
        overrides.extend(["SOLVER.MAX_ITER", str(int(config.max_iter))])
    
    if 'grad_clip_value' in config:
        overrides.extend(["SOLVER.CLIP_GRADIENTS.CLIP_VALUE", str(float(config.grad_clip_value))])
    
    # Warmup parameters
    if 'warmup_iters' in config:
        overrides.extend(["SOLVER.WARMUP_ITERS", str(int(config.warmup_iters))])
    
    if 'warmup_factor' in config:
        overrides.extend(["SOLVER.WARMUP_FACTOR", str(float(config.warmup_factor))])
    
    # Loss weights
    if 'dice_weight' in config:
        overrides.extend(["MODEL.MASK_FORMER.DICE_WEIGHT", str(float(config.dice_weight))])
    
    if 'mask_weight' in config:
        overrides.extend(["MODEL.MASK_FORMER.MASK_WEIGHT", str(float(config.mask_weight))])
    
    if 'class_weight' in config:
        overrides.extend(["MODEL.MASK_FORMER.CLASS_WEIGHT", str(float(config.class_weight))])
    
    return overrides


def main():
    run = wandb.init(project="FC-CLIP")
    cfg = run.config
    
    run_tag = f"sweep_{run.id}"
    
    cmd = [
        "python", "train_net.py",
        "--config-file", "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_train_lars_coco.yaml",
        "--num-gpus", "8", 
        "--wandb",
        "--tag", run_tag,
    ]
    
    # Add config overrides
    overrides = build_config_overrides(cfg)
    if overrides:
        cmd.extend(overrides)
    
    print("Executing:", " ".join(cmd), flush=True)
    
    # quick fix to avoid hanging wandb process
    wandb.finish()
    
    ret = subprocess.run(cmd).returncode
 
    run.finish(exit_code=ret)
    sys.exit(ret)


if __name__ == "__main__":
    main() 