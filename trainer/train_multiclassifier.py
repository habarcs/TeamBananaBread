"""
Module for training the MariaNet, resnet with multiple classifiers
"""
import itertools

import torch

from datasets.dataset_fetch import get_flowers_train_data_loader, get_flowers_test_data_loader, \
    get_flowers_val_data_loader
from models.multclassifiers1 import MultiClassifier
from trainer import DEVICE, train_loop, test_loop, RESULTS_DIR
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import wandb

from utils import EarlyStopper

BATCH_SIZE = 16
EPOCHS = 10


def main(wandb_active=True, load_name=None, early_stop=True):
    """
    Note: early stop only works on some datasets that have validation dataloader implemented
    """
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_loader, num_classes = \
        get_flowers_train_data_loader(transforms=transform_train, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = \
        get_flowers_test_data_loader(transforms=transform_test, batch_size=BATCH_SIZE, num_workers=4)
    validation_loader = \
        get_flowers_val_data_loader(transforms=transform_test, batch_size=BATCH_SIZE, num_workers=4)

    model = MultiClassifier(num_classes)
    num_classifiers = len(model.classifiers) + 1
    if load_name:
        model.load_state_dict(torch.load(RESULTS_DIR / load_name))

    model.to(DEVICE)
    if wandb_active:
        wandb.login()
        wandb.init(project="TeamBananaBread")
        wandb.watch(model)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(True),
        lr=0.002,
        momentum=0.9,
        weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0015)

    if early_stop:
        stopper = EarlyStopper(patience=1)
        for epoch in itertools.count(start=1):
            print(f"Epoch {epoch}\n-------------------------------")
            train_loop(train_loader, model, CELoss, optimizer, scheduler,
                       num_classifiers=num_classifiers, log=wandb_active)
            validation_loss = test_loop(validation_loader, model, CELoss, log=wandb_active, is_test=False)
            if stopper.early_stop(validation_loss):
                break
        test_loop(test_loader, model, CELoss, log=wandb_active)
        print("Done!")
    else:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader, model, CELoss, optimizer, scheduler, num_classifiers=num_classifiers,
                       log=wandb_active)
            test_loop(test_loader, model, CELoss, log=wandb_active)
        print("Done!")


if __name__ == '__main__':
    main()
