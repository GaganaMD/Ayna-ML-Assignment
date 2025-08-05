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

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

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

        colour_name = sample['colour']
        colour_idx = self.color2idx[colour_name]
        colour_tensor = torch.tensor(colour_idx, dtype=torch.long)

        return input_image, colour_tensor, output_image
