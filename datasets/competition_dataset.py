from pathlib import Path
from typing import List

from torch.utils.data import Dataset
from torchvision.io import read_image


class CompetitionTrainingDataset(Dataset):
    def __init__(self, data_directory, transforms=None):
        self.transforms = transforms
        if not isinstance(data_directory, Path):
            data_directory = Path(data_directory)
        self.image_directory = data_directory / "train" / "images"
        label_directory = data_directory / "train" / "labels"
        # TODO handle label read and fix classes ###
        self.labels: List[dict] = []
        self.classes = {label["label"] for label in self.labels}

    def __getitem__(self, idx):
        # TODO MAYBE HANDLE IMAGE NAME RESOLUTION
        image_name = self.labels[idx]["name"]
        # TODO handle label acquisition
        label = self.labels[idx]["label"]

        img = read_image(self.image_directory / image_name)
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


class CompetitionTestingDataset(Dataset):
    def __init__(self, data_directory, transforms=None):
        self.transforms = transforms
        if not isinstance(data_directory, Path):
            data_directory = Path(data_directory)
        self.image_directory = data_directory / "test" / "images"
        self.files = list(self.image_directory.iterdir())

    def __getitem__(self, idx):
        image_name = self.files[idx]
        img = read_image(self.image_directory / image_name)
        if self.transforms:
            img = self.transforms(img)
        return img, image_name

    def __len__(self):
        return len(self.files)
