from pathlib import Path
import torch
import torch.utils.data
from .torchvision_datasets import CocoDetection

from .dataset import build
from .dataset_fewshot import build as build_fewshot


# Meta-settings for few-shot object detection: base / novel category split
coco_base_class_ids = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

coco_novel_class_ids = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]

voc_base1_class_ids = [
    1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20
]

voc_novel1_class_ids = [
    3, 6, 10, 14, 18
]

voc_base2_class_ids = [
    2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20
]

voc_novel2_class_ids = [
    1, 5, 10, 13, 18
]

voc_base3_class_ids = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20
]

voc_novel3_class_ids = [
    4, 8, 14, 17, 18
]


def get_class_ids(dataset, type):
    if dataset == 'coco_base':
        if type == 'all':
            ids = (coco_base_class_ids + coco_novel_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return coco_base_class_ids
        elif type == 'novel':
            return coco_novel_class_ids
        else:
            raise ValueError
    if dataset == 'coco':
        if type == 'all':
            ids = (coco_base_class_ids + coco_novel_class_ids)
            ids.sort()
            return ids
        else:
            raise ValueError
    if dataset == 'voc_base1':
        if type == 'all':
            ids = (voc_base1_class_ids + voc_novel1_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base1_class_ids
        elif type == 'novel':
            return voc_novel1_class_ids
        else:
            raise ValueError
    if dataset == 'voc_base2':
        if type == 'all':
            ids = (voc_base2_class_ids + voc_novel2_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base2_class_ids
        elif type == 'novel':
            return voc_novel2_class_ids
        else:
            raise ValueError
    if dataset == 'voc_base3':
        if type == 'all':
            ids = (voc_base3_class_ids + voc_novel3_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base3_class_ids
        elif type == 'novel':
            return voc_novel3_class_ids
        else:
            raise ValueError
    if dataset == 'voc':
        if type == 'all':
            ids = (voc_base1_class_ids + voc_novel1_class_ids)
            ids.sort()
            return ids
        else:
            raise ValueError
    raise ValueError


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    assert image_set in ['train', 'val', 'fewshot'], "image_set must be 'train', 'val' or 'fewshot'."
    # For training set, need to perform base/novel category filtering.
    # For training set, we use dataset with support to construct meta-tasks
    if image_set == 'train':
        if args.dataset_file == 'coco':
            root = Path('data/coco')
            img_folder = root / "train2017"
            ann_file = root / "annotations" / 'instances_train2017.json'
            class_ids = coco_base_class_ids + coco_novel_class_ids
            class_ids.sort()
            return build(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=True)
        if args.dataset_file == 'coco_base':
            root = Path('data/coco')
            img_folder = root / "train2017"
            ann_file = root / "annotations" / 'instances_train2017.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=coco_base_class_ids, with_support=True)
        if args.dataset_file == 'voc':
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_trainval0712.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=list(range(1, 20+1)), with_support=True)
        if args.dataset_file == 'voc_base1':
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_trainval0712.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=voc_base1_class_ids, with_support=True)
        if args.dataset_file == 'voc_base2':
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_trainval0712.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=voc_base2_class_ids, with_support=True)
        if args.dataset_file == 'voc_base3':
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_trainval0712.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=voc_base3_class_ids, with_support=True)

    # For valid set, no need to perform base/novel category filtering.
    # This is because that evaluation should be performed on all images.
    # For valid set, we do not need support dataset.
    if image_set == 'val':
        if args.dataset_file in ['coco', 'coco_base']:
            root = Path('data/coco')
            img_folder = root / "val2017"
            ann_file = root / "annotations" / 'instances_val2017.json'
            class_ids = coco_base_class_ids + coco_novel_class_ids
            class_ids.sort()
            return build(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=False)
        if args.dataset_file in ['voc', 'voc_base1', 'voc_base2', 'voc_base3']:
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_test2007.json'
            return build(args, img_folder, ann_file, image_set, activated_class_ids=list(range(1, 20+1)), with_support=False)

    if image_set == 'fewshot':
        if args.dataset_file in ['coco', 'coco_base']:
            class_ids = coco_base_class_ids + coco_novel_class_ids
            class_ids.sort()
            return build_fewshot(args, image_set, activated_class_ids=class_ids, with_support=True)
        if args.dataset_file in ['voc', 'voc_base1', 'voc_base2', 'voc_base3']:
            return build_fewshot(args, image_set, activated_class_ids=list(range(1, 20+1)), with_support=True)

    raise ValueError(f'{image_set} of dataset {args.dataset_file}  not supported.')
