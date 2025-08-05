import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PolygonColorDataset(Dataset):
    def __init__(self, data_dir, json_path, color2idx, transform=None):
        self.data_dir = data_dir
        self.color2idx = color2idx
        self.transform = transforms.ToTensor() if transform is None else transform

        # Load the metadata
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_path = os.path.join(self.data_dir, 'inputs', sample['input_polygon'])
        output_path = os.path.join(self.data_dir, 'outputs', sample['output_image'])

        input_image = Image.open(input_path).convert("RGB")
        output_image = Image.open(output_path).convert("RGB")

        input_image = self.transform(input_image)
        output_image = self.transform(output_image)

        color_idx = torch.tensor(self.color2idx[sample['colour']], dtype=torch.long)
        return input_image, color_idx, output_image
