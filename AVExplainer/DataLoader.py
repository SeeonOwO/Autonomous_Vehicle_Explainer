import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from torchvision.transforms import transforms
import os
import json
import detectron2.data.transforms as T
from Explainer import *


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
        '''
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        '''
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (1280, 736))
        img_normalized = (img_resized - [103.53, 116.28, 123.675]) / [1.0, 1.0, 1.0]
        image = img_normalized.astype("float32").transpose(2, 0, 1)
        image = torch.as_tensor(image)

        actions = self.annotations[list(self.annotations.keys())[index]]['actions']
        reasons = self.annotations[list(self.annotations.keys())[index]]['reason']

        return image, torch.tensor(actions, dtype=torch.float32), torch.tensor(reasons, dtype=torch.float32), img_name
