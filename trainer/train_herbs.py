import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

from datasets.dataset_fetch import get_birds_train_data_loader, get_birds_test_data_loader
from models.HERBS import HERBS
from trainer import DEVICE, train_loop, test_loop, RESULTS_DIR

EPOCHS = 5
BATCH_SIZE = 16


def main(wandb_active=True, load_name=None):
    train_loader, num_classes = get_birds_train_data_loader(transforms=HERBS.transforms_train, batch_size=BATCH_SIZE)
    test_loader = get_birds_test_data_loader(transforms=HERBS.transforms_test, batch_size=BATCH_SIZE)
    model = HERBS(num_classes)
    if load_name:
        model.load_state_dict(torch.load(RESULTS_DIR / load_name))
    model.to(DEVICE)
    if wandb_active:
        wandb.login()
        wandb.init(project="TeamBananaBread")
        wandb.watch(model)

    # Now comes the training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7, last_epoch=-1)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer=optimizer, scheduler=None, log=wandb_active,
                   num_classifiers=5)
        test_loop(test_loader, model, loss_fn, log=wandb_active)
    print("Done!")


if __name__ == '__main__':
    main()
