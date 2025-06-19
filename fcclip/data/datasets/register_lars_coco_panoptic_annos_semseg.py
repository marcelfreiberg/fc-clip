import json
import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from . import openseg_classes

from detectron2.utils.file_io import PathManager


# Use the prompt-engineered categories as requested
LARS_COCO_CATEGORIES = openseg_classes.get_lars_coco_categories_with_prompt_eng()

_PREDEFINED_SPLITS_LARS_COCO_PANOPTIC = {
    "lars_coco_train_panoptic": (
        "panoptic_train",
        "annotations/panoptic_train.json",
        "panoptic_semseg_train_full",  # Use full semantic masks with all categories
    ),
    "lars_coco_val_panoptic": (
        "panoptic_val",
        "annotations/panoptic_val.json",
        "panoptic_semseg_val_full",  # Use full semantic masks with all categories
    ),
}


def get_metadata():
    meta = {}
    
    # Sort categories by original ID to ensure consistent ordering (same as preparation script)
    sorted_categories = sorted(LARS_COCO_CATEGORIES, key=lambda x: x["id"])
    
    # Split thing vs. stuff classes
    thing_classes = [k["name"] for k in sorted_categories if k["isthing"] == 1]
    thing_colors = [k["color"] for k in sorted_categories if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in sorted_categories if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in sorted_categories if k["isthing"] == 0]
    
    # For semantic evaluation, we need all classes in a unified list
    # Following COCO pattern: all classes get contiguous IDs [0, N-1]
    all_classes = [k["name"] for k in sorted_categories]
    all_colors = [k["color"] for k in sorted_categories]
    
    # Create unified contiguous ID mapping (like COCO does)
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_class_name = []
    
    # Assign contiguous IDs in sorted order
    for i, cat in enumerate(sorted_categories):
        if cat["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # For semantic evaluation, ALL classes go in stuff_dataset_id_to_contiguous_id
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        contiguous_id_to_class_name.append(cat["name"])
    
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = all_classes  # All classes for semantic evaluation
    meta["stuff_colors"] = all_colors    # All colors for semantic evaluation 
    meta["sem_stuff_classes"] = all_classes  # For text embeddings
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_class_name"] = contiguous_id_to_class_name
    meta["dataname"] = "lars_coco_val_panoptic"

    return meta


def load_lars_coco_panoptic_json(root, json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to "~/<root>/images/train" or "…/images/val"
        gt_dir (str): path to "~/<root>/<panoptic_root>"
        json_file (str): path to "~/<root>/annotations/panoptic_*.json"
        semseg_dir (str): path to "~/<root>/<panoptic_semseg_*>"
    Returns:
        list[dict]: Detectron2-format dicts
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # images end in .jpg
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        pan_seg_file = os.path.join(gt_dir, ann["file_name"])
        sem_seg_file = os.path.join(semseg_dir, ann["file_name"])

        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": pan_seg_file,
                "sem_seg_file_name": sem_seg_file,
                "segments_info": segments_info,
                "meta": {"dataname": meta["dataname"]},  # Required by model
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    first = ret[0]
    assert PathManager.isfile(first["file_name"]), first["file_name"]
    assert PathManager.isfile(first["pan_seg_file_name"]), first["pan_seg_file_name"]
    assert PathManager.isfile(first["sem_seg_file_name"]), first["sem_seg_file_name"]
    return ret


def register_lars_coco_panoptic_annos_sem_seg(
    root, name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root
):
    DatasetCatalog.register(
        name,
        lambda: load_lars_coco_panoptic_json(
            root, panoptic_json, image_root, panoptic_root, sem_seg_root, metadata
        ),
    )
    MetadataCatalog.get(name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        dataset_name=name,
        **metadata,
    )


def register_all_lars_coco_panoptic_annos_sem_seg(root):
    for prefix, (panoptic_root, panoptic_json, semantic_root) in _PREDEFINED_SPLITS_LARS_COCO_PANOPTIC.items():
        if "train" in prefix:
            image_root = os.path.join(root, "images/train")
        elif "val" in prefix:
            image_root = os.path.join(root, "images/val")
        else:
            raise ValueError(f"Unknown split prefix: {prefix}")

        # Update metadata dataname for this specific dataset
        metadata = get_metadata()
        metadata["dataname"] = prefix

        # Register both base dataset and _with_sem_seg version
        for name_suffix in ["", "_with_sem_seg"]:
            dataset_name = prefix + name_suffix
            if dataset_name not in DatasetCatalog:
                register_lars_coco_panoptic_annos_sem_seg(
                    root,
                    dataset_name,
                    metadata,
                    image_root,
                    os.path.join(root, panoptic_root),
                    os.path.join(root, panoptic_json),
                    os.path.join(root, semantic_root),
                )

_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
register_all_lars_coco_panoptic_annos_sem_seg(_root)