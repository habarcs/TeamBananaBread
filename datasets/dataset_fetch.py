"""
common module for defining dataloaders, the train dataloader also returns the number of classes of the dataset
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from datasets.cub200 import Cub2011
from utils import get_project_root
from datasets.competition_dataset import CompetitionTrainingDataset, CompetitionTestingDataset

DATA_SET_ROOT = get_project_root().parent / "data"  # being outside the project makes it easier to sync to remote


def get_fgvca_train_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    trainval_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="trainval",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform,
    )
    num_classes = len(set(trainval_data.classes))
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader, num_classes


def get_fgvca_test_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    test_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="test",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataloader


def get_flowers_train_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    trainval_data = datasets.Flowers102(
        root=DATA_SET_ROOT,
        split="train",
        download=True,
        transform=transforms,
        target_transform=target_transform,
    )
    num_classes = len(set(trainval_data._labels))
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader, num_classes


def get_flowers_test_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    test_data = datasets.Flowers102(
        root=DATA_SET_ROOT,
        split="test",
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataloader


def get_birds_train_data_loader(transforms=None, batch_size=64, num_workers=2):
    trainval_data = Cub2011(root=DATA_SET_ROOT,
                            train=True,
                            transform=transforms,
                            download=True)
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = 200
    return trainval_dataloader, num_classes


def get_birds_test_data_loader(transforms=None, batch_size=64, num_workers=2):
    test_data = Cub2011(root=DATA_SET_ROOT,
                        train=False,
                        transform=transforms,
                        download=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataloader


def get_comp_train_data_loader(data_directory, transforms=None, batch_size=64, num_workers=2):
    # TODO fix data directory
    trainval_data = CompetitionTrainingDataset(data_directory=data_directory, transforms=transforms)
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = len(trainval_data.classes)
    return trainval_dataloader, num_classes


def get_comp_test_data_loader(data_directory, transforms=None):
    # TODO fix data directory
    test_data = CompetitionTestingDataset(data_directory=data_directory, transforms=transforms)
    return test_data
