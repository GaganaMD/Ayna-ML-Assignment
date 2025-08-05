import torch
import os, json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PolygonColorDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load metadata
        with open(os.path.join(data_dir, 'data.json')) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Use the correct keys from your JSON
        input_path = os.path.join(self.data_dir, 'inputs', sample['input_polygon'])
        output_path = os.path.join(self.data_dir, 'outputs', sample['output_image'])

        input_image = Image.open(input_path).convert("RGB")
        output_image = Image.open(output_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

