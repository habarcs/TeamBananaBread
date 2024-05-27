"""
Trying out general image-classification network on fine-grained classification problems
"""
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

from datasets.dataset_fetch import get_fgvca_data_loader
from trainer import train_loop, test_loop
from utils import get_project_root

DATA_SET_ROOT = get_project_root() / "data"
ALEX_NET_OG_BATCH_SIZE = 128
ALEX_NET_OG_DROPOUT = 0.5
ALEX_NET_OG_WEIGHT_DECAY = 1e-4
ALEX_NET_OG_LEARNING_RATE = 1e-2  # it used decaying learning rate
ALEX_NET_OG_MOMENTUM = 0.9
EPOCHS = 90
NUMBER_OF_CLASSES = 100  # depends on the dataset, for FGVCA aircraft it is 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_transformer = models.AlexNet_Weights.DEFAULT.transforms()

trainval_dataloader, test_dataloader = get_fgvca_data_loader(transforms=image_transformer,
                                                             batch_size=ALEX_NET_OG_BATCH_SIZE)

# we load the alexnet model with pretrained weights
alex_net = models.alexnet(
    weights=models.AlexNet_Weights.DEFAULT
)

# here we set the weights of the model not trainable
for param in alex_net.parameters(recurse=True):
    param.requires_grad = False

# we want to use Alexnet feature extractor, but with a different classifier, that takes into account
# the classes we have, so we replace the original with a new one
alex_net.classifier = nn.Sequential(
    nn.Dropout(p=ALEX_NET_OG_DROPOUT),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=ALEX_NET_OG_DROPOUT),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, NUMBER_OF_CLASSES),
)

# Now comes the training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(alex_net.parameters(), lr=ALEX_NET_OG_LEARNING_RATE,
                            momentum=ALEX_NET_OG_MOMENTUM, weight_decay=ALEX_NET_OG_WEIGHT_DECAY)
alex_net.to(DEVICE)

for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(trainval_dataloader, alex_net, loss_fn, optimizer)
    test_loop(test_dataloader, alex_net, loss_fn)
print("Done!")
