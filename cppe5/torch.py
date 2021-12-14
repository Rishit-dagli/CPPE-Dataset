import os
from typing import Tuple

import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class Cppe5(torch.utils.data.Dataset):
    def __init__(self, root: str, annotation: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path))

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        annotations = {}
        annotations["boxes"] = boxes
        annotations["labels"] = labels
        annotations["image_id"] = img_id
        annotations["area"] = areas
        annotations["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, annotations

    def __len__(self) -> int:
        return len(self.ids)


def tensor_transform() -> torchvision.transforms.Compose:
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def data_loader(
    train_batch_size: int = 1,
    train_data_dir: str = "data/images",
    train_annotation_file: str = "data/annotations/train.json",
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:

    cppe5 = Cppe5(
        root=train_data_dir,
        annotation=train_annotation_file,
        transforms=tensor_transform(),
    )

    return torch.utils.data.DataLoader(
        cppe5,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
