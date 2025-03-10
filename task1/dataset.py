import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class DatasetManager:
    @classmethod
    def get_dataset_loader(
        cls, data_path, class_path, image_dir, batch_size, is_training=True
    ):
        if is_training:
            data_loader = DataLoader(
                Dataset(data_path, image_dir),
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            data_loader = DataLoader(
                Dataset(data_path, image_dir),
                batch_size=batch_size,
                shuffle=False,
            )
        with open(class_path, "r") as f:
            class_lines = f.readlines()
            class_lines = [line.strip().split(" ") for line in class_lines]
        class_mapping = {int(line[0]): line[1] for line in class_lines}

        return data_loader, class_mapping


class Dataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize([128, 128]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.data = pd.read_csv(csv_path, header=None)
        self.image_paths = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1].astype(int).values

        label_counts = self.data.iloc[:, 1].value_counts().to_dict()
        total_samples = len(self.data)
        total_classes = len(np.unique(self.labels))
        self.weights = torch.tensor(
            [
                total_samples / (label_counts[label] * total_classes)
                for label in range(total_classes)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = self.transform(Image.open(image_path))
        label = self.labels[index]
        return image, label
