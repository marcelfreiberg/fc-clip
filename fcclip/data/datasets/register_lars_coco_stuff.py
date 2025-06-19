import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

# Use the prompt-engineered categories as requested
LARS_COCO_CATEGORIES = openseg_classes.get_lars_coco_categories_with_prompt_eng()


def _get_lars_coco_stuff_meta():
    # Sort categories by original ID to ensure consistent ordering (same as preparation script)
    sorted_categories = sorted(LARS_COCO_CATEGORIES, key=lambda x: x["id"])
    
    # Get all category IDs and names in sorted order
    stuff_ids = [k["id"] for k in sorted_categories]
    
    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, N], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in sorted_categories]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def load_lars_coco_semantic_with_meta(image_dir, sem_seg_dir, meta):
    """Load semantic segmentation dataset with required meta key"""
    # Use built-in loader and add meta key
    dataset_dicts = load_sem_seg(sem_seg_dir, image_dir, gt_ext="png", image_ext="jpg")
    
    # Add required meta key to each item
    for item in dataset_dicts:
        item["meta"] = {"dataname": meta["dataname"]}
    
    return dataset_dicts


def register_all_lars_coco_stuff(root):
    meta = _get_lars_coco_stuff_meta()

    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "panoptic_semseg_train_full"),
        ("val", "images/val", "panoptic_semseg_val_full"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"lars_coco_{name}_sem_seg"
        
        # Create a copy of meta and add the dataname
        newmeta = meta.copy()
        newmeta['dataname'] = all_name
        
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir, m=newmeta: load_lars_coco_semantic_with_meta(x, y, m),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            dataset_name=name,
            **newmeta,
        )


_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
register_all_lars_coco_stuff(_root) 