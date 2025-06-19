import json
import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from . import openseg_classes
import copy

# Use the prompt-engineered categories as requested but filter for things only
LARS_COCO_ALL_CATEGORIES = openseg_classes.get_lars_coco_categories_with_prompt_eng()
# Filter to only "thing" categories (objects) for instance segmentation
LARS_COCO_CATEGORIES = [x for x in LARS_COCO_ALL_CATEGORIES if x["isthing"] == 1]

_PREDEFINED_SPLITS = {
    "lars_coco_train": (
        "images/train",
        "annotations/panoptic_train.json",
    ),
    "lars_coco_val": (
        "images/val", 
        "annotations/panoptic_val.json",
    ),
}


def _get_lars_coco_instances_meta():
    # Sort categories by original ID to ensure consistent ordering
    sorted_categories = sorted(LARS_COCO_CATEGORIES, key=lambda x: x["id"])
    
    thing_ids = [k["id"] for k in sorted_categories]
    # Mapping from the original category id to contiguous id in [0, #things)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in sorted_categories]
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def load_lars_coco_instance_json(json_file, image_root, dataset_name):
    """
    Load LARS COCO instance annotations from panoptic JSON format.
    Extract only "thing" classes for instance segmentation.
    """
    from pycocotools import mask as maskUtils
    
    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    # Get metadata for ID mapping
    meta = _get_lars_coco_instances_meta()
    
    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        
        # Find corresponding image info
        image_info = None
        for img in json_info["images"]:
            if img["id"] == image_id:
                image_info = img
                break
        
        if image_info is None:
            continue
            
        # images end in .jpg
        image_file = os.path.join(image_root, os.path.splitext(ann["file_name"])[0] + ".jpg")
        
        # Extract only thing instances from segments_info
        instances = []
        for segment_info in ann["segments_info"]:
            original_cat_id = segment_info["category_id"]
            
            # Check if this is a "thing" category
            category_info = None
            for cat in LARS_COCO_ALL_CATEGORIES:
                if cat["id"] == original_cat_id:
                    category_info = cat
                    break
            
            if category_info is None or category_info["isthing"] != 1:
                continue  # Skip stuff categories
            
            # Convert to contiguous ID
            contiguous_id = meta["thing_dataset_id_to_contiguous_id"][original_cat_id]
            
            # Create instance annotation
            instances.append({
                "id": segment_info["id"],
                "category_id": contiguous_id,
                "area": segment_info["area"],
                "bbox": segment_info["bbox"],
                "iscrowd": 0,
            })
        
        if instances:  # Only add if there are thing instances
            ret.append({
                "file_name": image_file,
                "image_id": image_id,
                "height": image_info["height"],
                "width": image_info["width"],
                "annotations": instances,
            })

    return ret


def register_all_lars_coco_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        DatasetCatalog.register(
            key,
            lambda j=json_file, i=image_root: load_lars_coco_instance_json(
                os.path.join(root, j), os.path.join(root, i), key
            ),
        )
        MetadataCatalog.get(key).set(
            json_file=os.path.join(root, json_file),
            image_root=os.path.join(root, image_root),
            evaluator_type="coco",
            **_get_lars_coco_instances_meta(),
        )


_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
register_all_lars_coco_instance(_root) 