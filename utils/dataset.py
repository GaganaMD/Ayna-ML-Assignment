import torch
import os, json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PolygonColorDataset(Dataset):
    def __init__(self, root, json_path, color2idx, transform=None):
        super().__init__()
        with open(json_path, 'r') as f:
            self.mapping = json.load(f)
        self.root = root
        self.color2idx = color2idx
        self.transform = transform

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        rec = self.mapping[idx]
        img_x = Image.open(os.path.join(self.root, "inputs", rec['input'])).convert('L')
        img_y = Image.open(os.path.join(self.root, "outputs", rec['output'])).convert('RGB')
        color = rec['color']
        # Preprocess
        if self.transform:
            img_x = self.transform(img_x)
            img_y = self.transform(img_y)
        else:
            img_x = np.array(img_x, dtype=np.float32) / 255.0
            img_x = np.expand_dims(img_x, axis=0)
            img_y = np.array(img_y, dtype=np.float32) / 255.0
            img_y = img_y.transpose(2,0,1)
            img_x, img_y = torch.tensor(img_x), torch.tensor(img_y)
        return img_x, self.color2idx[color], img_y
