from pathlib import Path

from datasets.dataset import DetectionDataset
import datasets.transforms as T
from util.misc import get_local_rank, get_local_size


def make_transforms():
    """
    Transforms for query images during the few-shot fine-tuning stage.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomColorJitter(p=0.3333),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1152),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1152),
            ])
        ),
        normalize,
    ])


def make_support_transforms():
    """
    Transforms for support images during the few-shot fine-tuning stage.
    For transforms for support images during the base training stage, please check dataset.py
    For transforms for support images during inference, please check dataset_support.py
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomColorJitter(p=0.25),
        T.RandomSelect(
            T.RandomResize(scales, max_size=672),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=672),
            ])
        ),
        normalize,
    ])


def build(args, image_set, activated_class_ids, with_support=True):
    assert image_set == "fewshot"
    activated_class_ids.sort()

    if args.dataset_file in ['coco_base']:
        root = Path('data/coco_fewshot')
        img_folder = root.parent / 'coco' / "train2017"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base1":
        root = Path('data/voc_fewshot_split1')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base2":
        root = Path('data/voc_fewshot_split2')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base3":
        root = Path('data/voc_fewshot_split3')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    raise ValueError
