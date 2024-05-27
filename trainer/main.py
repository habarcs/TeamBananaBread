import torchvision.models.swin_transformer
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from datasets.dataset_fetch import get_fgvca_data_loader
from models.HERBS import HERBS
from trainer import DEVICE, train_loop, test_loop

EPOCHS = 90
BATCH_SIZE = 32


def main():
    train_loader, test_loader = get_fgvca_data_loader(transforms=HERBS.transforms(), batch_size=BATCH_SIZE)
    model = HERBS(len(train_loader.dataset.classes))
    model.to(DEVICE)

    # Now comes the training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=False)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-7, last_epoch=-1)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer=optimizer, scheduler=scheduler)
        test_loop(test_loader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    main()

torchvision.models.swin_transformer.swin_b()
