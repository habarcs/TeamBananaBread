"""
Main module for training the swin model by itself
"""
import timm
import wandb
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from competition.submit import competition_test_loop
from datasets.dataset_fetch import get_comp_train_data_loader, get_comp_test_data_loader
from trainer import DEVICE, train_loop

EPOCHS = 300
BATCH_SIZE = 8

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])
transform_test = transforms.Compose([
    transforms.Resize(510),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])


def main(wandb_active=True):
    train_loader, num_classes = get_comp_train_data_loader(data_directory="/home/disi/competition_data",
                                                           transforms=transform_train, batch_size=BATCH_SIZE)
    test_loader = \
        get_comp_test_data_loader(data_directory="/home/disi/competition_data", transforms=transform_test)

    model = timm.create_model('swin_base_patch4_window12_384', num_classes=num_classes, pretrained=True)

    model.to(DEVICE)
    if wandb_active:
        wandb.login()
        wandb.init(project="TeamBananaBread")
        wandb.watch(model)

    # Now comes the training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-7)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-7)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer=optimizer, scheduler=scheduler, log=wandb_active)
        competition_test_loop(model, test_loader, train_loader.dataset.classes)
    print("Done!")


if __name__ == '__main__':
    main()
