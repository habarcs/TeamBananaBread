from torch.utils.data import DataLoader
from torchvision import datasets
from datasets.cub200 import Cub2011
from utils import get_project_root

DATA_SET_ROOT = get_project_root() / "data"


def get_fgvca_train_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    trainval_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="trainval",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform,
    )
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader


def get_fgvca_test_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    test_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="test",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    num_classes = len(set(test_data.classes))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_dataloader, num_classes


def get_flowers_train_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    trainval_data = datasets.Flowers102(
        root=DATA_SET_ROOT,
        split="train",
        download=True,
        transform=transforms,
        target_transform=target_transform,
    )
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader


def get_flowers_test_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    test_data = datasets.Flowers102(
        root=DATA_SET_ROOT,
        split="test",
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    num_classes = len(set(test_data._labels))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_dataloader, num_classes


def get_birds_train_data_loader(transforms=None, batch_size=64, num_workers=2):
    trainval_data = Cub2011(root=DATA_SET_ROOT,
                            train=True,
                            transform=transforms,
                            download=True)
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader


def get_birds_test_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    test_data = Cub2011(root=DATA_SET_ROOT,
                        train=False,
                        transform=transforms,
                        download=True)
    num_classes = 200
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_dataloader, num_classes
