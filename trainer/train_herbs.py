from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

from datasets.dataset_fetch import get_fgvca_train_data_loader, get_fgvca_test_data_loader
from models.HERBS import HERBS
from trainer import DEVICE, train_loop, test_loop
EPOCHS = 300
BATCH_SIZE = 32


def main(wandb_active=True):
    train_loader = get_fgvca_train_data_loader(transforms=HERBS.transforms(), batch_size=BATCH_SIZE)
    test_loader, num_classes = get_fgvca_test_data_loader(transforms=HERBS.transforms(), batch_size=BATCH_SIZE)
    model = HERBS(num_classes)
    model.to(DEVICE)
    if wandb_active:
        wandb.login()
        wandb.init(project="TeamBananaBread")
        wandb.watch(model)

    # Now comes the training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-7)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7, last_epoch=-1)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer=optimizer, scheduler=scheduler, log=wandb_active)
        test_loop(test_loader, model, loss_fn, log=wandb_active)
    print("Done!")


if __name__ == '__main__':
    main()
