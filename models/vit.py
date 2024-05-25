"""
Trying out VIT model
"""

import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_project_root

DATA_SET_ROOT = get_project_root() / "data"
VIT_NET_OG_BATCH_SIZE = 512
VIT_NET_OG_WEIGHT_DECAY = 1e-4
VIT_NET_OG_LEARNING_RATE = 1e-2
# VIT_OG_DROPOUT = 0.5
VIT_NET_OG_MOMENTUM = 0.9
EPOCHS = 40
NUMBER_OF_CLASSES = 100  # depends on the dataset, for FGVCA aircraft it is 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

label_transformer = transforms.Lambda(lambda y: torch.zeros(NUMBER_OF_CLASSES, dtype=torch.float)
                                      .scatter_(0, torch.tensor(y), value=1))  # one-hot encoding

default_transforms = models.ViT_B_32_Weights.DEFAULT.transforms()

# Changing input resolution for fine-tuning
new_resolution = (384, 384)

# Define the transform pipeline (using default transforms of vit)
new_transforms = transforms.Compose([
    transforms.Resize(new_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    # Resize step with specified interpolation
    transforms.CenterCrop(224),  # Center crop
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with provided mean and std
])
trainval_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="trainval",
    annotation_level="variant",
    download=True,
    transform=new_transforms,
    target_transform=label_transformer
)
trainval_dataloader = DataLoader(trainval_data, batch_size=VIT_NET_OG_BATCH_SIZE, shuffle=True)

test_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="test",
    annotation_level="variant",
    download=True,
    transform=new_transforms,
    target_transform=None
)
test_dataloader = DataLoader(test_data, batch_size=VIT_NET_OG_BATCH_SIZE, shuffle=False)

# we load vit model with pretrained weights
vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)

# here we set the weights of the model not trainable
for param in vit.parameters(recurse=True):
    param.requires_grad = False

# for layer in vit.encoder.layers.encoder_layer_11.mlp:  # Unfreeze the last 4 layers
#     for param in layer.parameters():
#         param.requires_grad = True
# for param in vit.encoder.ln.parameters():  # Unfreeze the last 4 layers
#         param.requires_grad = True


num_features = vit.heads.head.in_features  # 768

# #we follow what is suggested in the paper: https://arxiv.org/pdf/2010.11929
# #so having 1 linear layer with zero weights
# vit.heads = nn.Sequential(
#    nn.Linear(num_features, NUMBER_OF_CLASSES),
#    )

# # try more layers
vit.heads = nn.Sequential(
    nn.Linear(in_features=num_features, out_features=512),
    nn.Dropout(0.3),
    nn.GELU(),
    nn.Linear(in_features=512, out_features=NUMBER_OF_CLASSES)
)

# Initialize the weights to zero
nn.init.constant_(vit.heads[0].weight, 0.0)

# Initialize the bias to zero
if vit.heads[0].bias is not None:
    nn.init.constant_(vit.heads[0].bias, 0.0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=VIT_NET_OG_LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


vit.to(DEVICE)

for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(trainval_dataloader, vit, loss_fn, optimizer)
    scheduler.step()
    test_loop(test_dataloader, vit, loss_fn)
print("Done!")
