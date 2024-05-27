from torch.utils.data import DataLoader
from torchvision import datasets

from utils import get_project_root

DATA_SET_ROOT = get_project_root() / "data"


def get_fgvca_data_loader(transforms=None, target_transform=None, batch_size=64, num_workers=2):
    trainval_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="trainval",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform,
    )
    trainval_dataloader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = datasets.FGVCAircraft(
        root=DATA_SET_ROOT,
        split="test",
        annotation_level="variant",
        download=True,
        transform=transforms,
        target_transform=target_transform
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainval_dataloader, test_dataloader
