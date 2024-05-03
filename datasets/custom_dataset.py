from __future__ import division
import sys
import json
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF  
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from datasets.base_dataset import BaseDataset, BaseTransform
from datasets.transforms import RandomColorJitter

  
def save_visualization(image, img_box, boxes, file_name):
    # Draw bounding boxes on the image
    for box in boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Combine the original image and the masked image
    combined_image = np.concatenate((image, img_box), axis=1)

    # Save the visualization to the "vis" folder
    vis_dir = "vis"
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, file_name)
    cv2.imwrite(save_path, combined_image)
def build_custom_dataloader(cfg, training, distributed=True):
    rank = dist.get_rank()

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])

    if training:
        hflip = cfg.get("hflip", False)
        vflip = cfg.get("vflip", False)
        rotate = cfg.get("rotate", False)
        gamma = cfg.get("gamma", False)
        gray = cfg.get("gray", False)
        transform_fn = BaseTransform(
            cfg["input_size"], hflip, vflip, rotate, gamma, gray
        )
    else:
        transform_fn = BaseTransform(
            cfg["input_size"], False, False, False, False, False
        )

    if training and cfg.get("colorjitter", None):
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])
    else:
        colorjitter_fn = None

    if rank == 0:
        print("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        cfg["img_dir"],
        cfg["density_dir"],
        cfg["meta_file"],
        cfg["shot"],
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        density_dir,
        meta_file,
        shot,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.shot = shot
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        if isinstance(meta_file, str):
            meta_file = [meta_file]
        self.metas = []
        for _meta_file in meta_file:
            with open(_meta_file, "r+") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        # read img
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        img_box_path=os.path.join(os.path.dirname(self.img_dir),'img_box', img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)
        # get boxes, h, w
        boxes = meta["boxes"]
        if self.shot:
            boxes = boxes[: self.shot]
        img_box=cv2.imread(img_box_path)
        img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
        img_box[img_box>0]=255
        
             
        # transform
        if self.transform_fn:
            image, img_box,density, boxes, _ = self.transform_fn(
                image, img_box,density, boxes, [], (height, width)
            )
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
            img_box = self.colorjitter_fn(img_box)
        image = transforms.ToTensor()(image)
        img_box = transforms.ToTensor()(img_box)
        density = transforms.ToTensor()(density)
        boxes = torch.tensor(boxes, dtype=torch.float64)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        return {
            "filename": img_name,
            "img_box":img_box,
            "height": height,
            "width": width,
            "image": image,
            "density": density,
            "boxes": boxes,
        }
