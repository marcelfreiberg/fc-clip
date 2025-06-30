import sys
import subprocess
import wandb


def build_opts(config):
    """Return list of key=value strings for --opts."""
    opts = []

    # thresholds
    opts.append("MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD")
    opts.append(str(float(config.object_thr)))
    opts.append("MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD")
    opts.append(str(float(config.overlap_thr)))

    return opts


def main():
    run = wandb.init(project="fc-clip")
    cfg = run.config

    run_tag = f"inference_sweep_{run.id}"
    
    cmd = [
        "python", "train_net.py",
        "--config-file", "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_lars_coco.yaml",
        "--num-gpus", "8",
        "--eval-only",
        "--wandb",
        "--tag", run_tag,
    ]

    cmd.extend(build_opts(cfg))

    print("Executing:", " ".join(cmd), flush=True)
    
    # quick fix to avoid hanging wandb process
    wandb.finish()

    ret = subprocess.run(cmd).returncode

    run.finish(exit_code=ret)
    sys.exit(ret)


if __name__ == "__main__":
    main() 