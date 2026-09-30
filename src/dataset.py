"""Read one labeled face image at a time from a split manifest."""

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class FaceQualityDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int = 224, horizontal_flip: bool = False,
                 normalize_imagenet: bool = False):
        self.csv_path = Path(csv_path).resolve()
        self.root = self.csv_path.parents[1]
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.normalize_imagenet = normalize_imagenet
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows or set(self.rows[0]) != {"image_id", "image_path", "quality_score"}:
            raise ValueError(f"Invalid or empty split CSV: {self.csv_path}")
        for row in self.rows:
            relative = Path(row["image_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "train":
                raise ValueError(f"Unsafe image path: {relative}")
            row["full_path"] = self.root / relative
            row["label"] = float(row["quality_score"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        with Image.open(row["full_path"]) as source:
            image = source.convert("RGB")
            # Keep the whole face and its aspect ratio; pad to a square.
            image = ImageOps.pad(image, (self.image_size, self.image_size),
                                 method=Image.Resampling.BILINEAR, color=(128, 128, 128))
            if self.horizontal_flip and torch.rand(()) < 0.5:
                image = ImageOps.mirror(image)
            tensor = TF.to_tensor(image)
            if self.normalize_imagenet:
                tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225))
        label = torch.tensor(row["label"], dtype=torch.float32)
        return tensor, label, Path(row["image_path"]).name
