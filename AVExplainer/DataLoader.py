import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
import os
import json


class CustomDataset(Dataset):
    def __init__(self, data_dir, annotations, transform=None):
        self.data_dir = data_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = list(self.annotations.keys())[index]
        img_path = os.path.join(self.data_dir, list(self.annotations.keys())[index] + ".jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        actions = self.annotations[list(self.annotations.keys())[index]]['actions']
        reasons = self.annotations[list(self.annotations.keys())[index]]['reason']

        return img, torch.tensor(actions, dtype=torch.float32), torch.tensor(reasons, dtype=torch.float32), img_name
