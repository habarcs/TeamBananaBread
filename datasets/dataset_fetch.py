"""
This python package defines the dataloaders used for testing and training
,.boygoyg
"""
from utils import get_project_root

from torchvision import datasets as built_in_datasets
from torch.utils.data.dataset import Dataset, ConcatDataset

DATA_DIR = get_project_root() / "data"


def get_dataset() -> Dataset:
    """
    Creates a dataset that is a combination of multiple online datasets.
    :return: concatenated dataset
    """

    fgvca = built_in_datasets.FGVCAircraft(
        root=DATA_DIR,
        download=True
    )
    flowers = built_in_datasets.Flowers102(
        root=DATA_DIR,
        download=True
    )
    food = built_in_datasets.Food101(
        root=DATA_DIR,
        download=True
    )

    return ConcatDataset([fgvca, flowers, food])


if __name__ == '__main__':
    get_dataset()
