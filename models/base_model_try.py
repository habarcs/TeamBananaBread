"""
Trying out general image-classification network on fine-grained classification problems
"""
from torch import nn
from torchvision import models, datasets
from torch.utils.data import DataLoader

from utils import get_project_root

DATA_SET_ROOT = get_project_root() / "data"
ALEX_NET_OG_BATCH_SIZE = 128
ALEX_NET_OG_DROPOUT = 0.5

train_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="train",
    annotation_level="variant",
    download=True,
    transform=None,
    target_transform=None
)
train_dataloader = DataLoader(train_data, batch_size=ALEX_NET_OG_BATCH_SIZE, shuffle=True)

val_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="val",
    annotation_level="variant",
    download=True,
    transform=None,
    target_transform=None
)
val_dataloader = DataLoader(val_data, batch_size=ALEX_NET_OG_BATCH_SIZE, shuffle=True)

test_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="test",
    annotation_level="variant",
    download=True,
    transform=None,
    target_transform=None
)
test_dataloader = DataLoader(test_data, batch_size=ALEX_NET_OG_BATCH_SIZE, shuffle=True)


# we load the alexnet model with pretrained weights
alex_net = models.alexnet(
    weights=models.AlexNet_Weights.DEFAULT
)

# we want to use Alexnet feature extractor, but with a different classifier, that takes into account
# the classes we have, so we replace the original with a new one
alex_net.classifier = nn.Sequential(
    nn.Dropout(p=ALEX_NET_OG_DROPOUT),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=ALEX_NET_OG_DROPOUT),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, len(train_data.classes)),
)

# here we set the other parts of the model not trainable, so they stay the same
# and only the new classificator is changed
alex_net.features.train(False)
alex_net.avgpool.train(False)
