import os
import random
from PIL import Image
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class DetectionDataset(TvCocoDetection):
    def __init__(self, args, img_folder, ann_file, transforms, support_transforms, return_masks, activated_class_ids,
                 with_support, cache_mode=False, local_rank=0, local_size=1):
        super(DetectionDataset, self).__init__(img_folder, ann_file, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.with_support = with_support
        self.activated_class_ids = activated_class_ids
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        """
        If with_support = True, this dataset will also produce support images and support targets.
        with_support should be set to True for training, and should be set to False for inference.
          * During training, support images are sampled along with query images in this dataset.
          * During inference, support images are sampled from dataset_support.py
        """
        if self.with_support:
            self.NUM_SUPP = args.total_num_support
            self.NUM_MAX_POS_SUPP = args.max_pos_support
            self.support_transforms = support_transforms
            self.build_support_dataset(ann_file)

    def __getitem__(self, idx):
        img, target = super(DetectionDataset, self).__getitem__(idx)
        target = [anno for anno in target if anno['category_id'] in self.activated_class_ids]
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.with_support:
            support_images, support_class_ids, support_targets = self.sample_support_samples(target)
            return img, target, support_images, support_class_ids, support_targets
        else:
            return img, target

    def build_support_dataset(self, ann_file):
        self.anns_by_class = {i: [] for i in self.activated_class_ids}
        coco = COCO(ann_file)
        for classid in self.activated_class_ids:
            annIds = coco.getAnnIds(catIds=classid)
            for annId in annIds:
                ann = coco.loadAnns(annId)[0]
                if 'area' in ann:
                    if ann['area'] < 5.0:
                        continue
                if 'ignore' in ann:
                    if ann['ignore']:
                        continue
                if 'iscrowd' in ann:
                    if ann['iscrowd'] == 1:
                        continue
                ann['image_path'] = coco.loadImgs(ann['image_id'])[0]['file_name']
                self.anns_by_class[classid].append(ann)

    def sample_support_samples(self, target):
        positive_labels = target['labels'].unique()
        num_positive_labels = positive_labels.shape[0]
        positive_labels_list = positive_labels.tolist()
        negative_labels_list = list(set(self.activated_class_ids) - set(positive_labels_list))

        # Positive labels in a batch < TRAIN_NUM_POSITIVE_SUPP: we include additional labels as negative samples
        if num_positive_labels <= self.NUM_MAX_POS_SUPP:
            sampled_labels_list = positive_labels_list
            sampled_labels_list += random.sample(negative_labels_list, k=self.NUM_SUPP - num_positive_labels)
        # Positive labels in a batch > TRAIN_NUM_POSITIVE_SUPP: remove some positive labels.
        else:
            sampled_positive_labels_list = random.sample(positive_labels_list, k=self.NUM_MAX_POS_SUPP)
            sampled_negative_labels_list = random.sample(negative_labels_list, k=self.NUM_SUPP - self.NUM_MAX_POS_SUPP)
            sampled_labels_list = sampled_positive_labels_list + sampled_negative_labels_list
            # -----------------------------------------------------------------------
            # NOTE: There is no need to filter gt info at this stage.
            #       Filtering is done when formulating the episodes.
            # -----------------------------------------------------------------------

        support_images = []
        support_targets = []
        support_class_ids = []
        for class_id in sampled_labels_list:
            i = random.randint(0, len(self.anns_by_class[class_id]) - 1)
            support_target = self.anns_by_class[class_id][i]
            support_target = {'image_id': class_id, 'annotations': [support_target]}  # Actually it is class_id for key 'image_id' here
            support_image_path = os.path.join(self.root, self.anns_by_class[class_id][i]['image_path'])
            support_image = Image.open(support_image_path).convert('RGB')
            support_image, support_target = self.prepare(support_image, support_target)
            if self.support_transforms is not None:
                org_support_target, org_support_image = support_target, support_image
                while True:
                    support_image, support_target = self.support_transforms(org_support_image, org_support_target)
                    # Make sure the object is not deleted after transforms, and it is not too small (mostly cut off)
                    if support_target['boxes'].shape[0] == 1 and support_target['area'] >= org_support_target['area'] / 5.0:
                        break
            support_images.append(support_image)
            support_targets.append(support_target)
            support_class_ids.append(class_id)
        return support_images, torch.as_tensor(support_class_ids), support_targets


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_transforms(image_set):
    """
    Transforms for query images.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
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

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1152),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_support_transforms():
    """
    Transforms for support images during the training phase.
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


def build(args, img_folder, ann_file, image_set, activated_class_ids, with_support):
    return DetectionDataset(args, img_folder, ann_file,
                            transforms=make_transforms(image_set),
                            support_transforms=make_support_transforms(),
                            return_masks=False,
                            activated_class_ids=activated_class_ids,
                            with_support=with_support,
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size())
