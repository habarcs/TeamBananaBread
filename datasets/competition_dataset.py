from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_image


class CompetitionTrainingDataset(Dataset):
    def __init__(self, data_directory, transforms=None):
        self.transforms = transforms
        if not isinstance(data_directory, Path):
            data_directory = Path(data_directory)
        self.train_dir = data_directory / "train"
        self.labels = []
        self.classes = list({clx.name for clx in self.train_dir.iterdir()})
        for clx in self.train_dir.iterdir():
            for image in clx.iterdir():
                self.labels.append((image.name, clx.name))

    def __getitem__(self, idx):
        image_name, class_name = self.labels[idx]
        class_id = self.classes[self.classes.index(class_name)].split('_')[0]

        img = read_image(self.train_dir / class_name / image_name)
        if self.transforms:
            img = self.transforms(img)
        return img, class_id

    def __len__(self):
        return len(self.labels)


class CompetitionTestingDataset(Dataset):
    def __init__(self, data_directory, transforms=None):
        self.transforms = transforms
        if not isinstance(data_directory, Path):
            data_directory = Path(data_directory)
        self.test_directory = data_directory / "test"
        self.files = list(self.test_directory.iterdir())

    def __getitem__(self, idx):
        image_name = self.files[idx]
        img = read_image(self.test_directory / image_name)
        if self.transforms:
            img = self.transforms(img)
        return img, image_name.stem

    def __len__(self):
        return len(self.files)
