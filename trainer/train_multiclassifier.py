from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import wandb

from datasets.dataset_fetch import get_fgvca_train_data_loader, get_fgvca_test_data_loader
from models.multclassifiers1 import build_model
from trainer import DEVICE, train_loop, test_loop

BATCH_SIZE = 16
EPOCHS = 5


def main():
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
    train_loader = get_fgvca_train_data_loader(transforms=transform_train, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = get_fgvca_test_data_loader(transforms=transform_test, batch_size=BATCH_SIZE, num_workers=4)
    model = build_model(number_of_classes=len(train_loader.dataset.classes), device=DEVICE)
    wandb.login()
    wandb.init(project="TeamBananaBread")
    wandb.watch(model)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.classifier_concat.parameters(), 'lr': 0.002},
        {'params': model.conv_block1.parameters(), 'lr': 0.002},
        {'params': model.classifier1.parameters(), 'lr': 0.002},
        {'params': model.conv_block2.parameters(), 'lr': 0.002},
        {'params': model.classifier2.parameters(), 'lr': 0.002},
        {'params': model.conv_block3.parameters(), 'lr': 0.002},
        {'params': model.classifier3.parameters(), 'lr': 0.002},
        {'params': model.Features.parameters(), 'lr': 0.0002}  # not used as resnet is not trained

    ],
        momentum=0.9,
        weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_loader, model, CELoss, optimizer, scheduler, num_classifiers=4, log=True)
        test_loop(test_loader, model, CELoss, log=True)
    print("Done!")


if __name__ == '__main__':
    main()
