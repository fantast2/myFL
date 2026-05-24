from PIL import Image
import torch
from torchvision.datasets import SVHN


class CustomSVHN(SVHN):
    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super().__init__(root, split, transform, target_transform, download)
        self.targets = self.labels

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target


class IndustrialImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples          # [(img_path, label), ...]
        self.transform = transform
        self.targets = [y for _, y in samples]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label