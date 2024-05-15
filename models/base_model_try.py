"""
Trying out general image-classification network on fine-grained classification problems
"""
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

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

label_transformer = transforms.Lambda(lambda y: torch.zeros(NUMBER_OF_CLASSES, dtype=torch.float)
                                      .scatter_(0, torch.tensor(y), value=1))

image_transformer = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainval_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="trainval",
    annotation_level="variant",
    download=True,
    transform=image_transformer,
    target_transform=label_transformer
)
trainval_dataloader = DataLoader(trainval_data, batch_size=ALEX_NET_OG_BATCH_SIZE, shuffle=True)

test_data = datasets.FGVCAircraft(
    root=DATA_SET_ROOT,
    split="test",
    annotation_level="variant",
    download=True,
    transform=image_transformer,
    target_transform=None
)
test_dataloader = DataLoader(test_data, batch_size=ALEX_NET_OG_BATCH_SIZE, shuffle=True)

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


# this is copied from pytorch tutorial, seems general enough
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X.to(DEVICE)
        y.to(DEVICE)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# this is copied from pytorch tutorial, seems general enough
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X.to(DEVICE)
            y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
