import timm, pandas
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import wandb

from competition.submit import competition_test_loop
from datasets.dataset_fetch import get_fgvca_train_data_loader, get_fgvca_test_data_loader, get_birds_test_data_loader, \
    get_birds_train_data_loader, get_comp_train_data_loader, get_comp_test_data_loader
from trainer import DEVICE, train_loop, test_loop
from torchvision import transforms
print(timm.list_models('*swin*'))
EPOCHS = 300
BATCH_SIZE = 8

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])
transform_test = transforms.Compose([
    transforms.Resize(510), #first resize of 256x256
    transforms.CenterCrop(384), #second takes(cut) square of 224x224 as required by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])


def main(wandb_active=True):
    train_loader, num_classes = get_comp_train_data_loader(data_directory="/home/disi/competition_data", transforms=transform_train, batch_size=BATCH_SIZE)
    test_loader = \
        get_comp_test_data_loader(data_directory="/home/disi/competition_data", transforms=transform_test)

    model2 = timm.create_model('swin_tiny_patch4_window7_224', num_classes = num_classes, pretrained=True)

    #89% in 10 epochs [Resize(450) test images followed by CenterCrop(384), lr=1e-5, wd=1e-7, AdamW]
    model = timm.create_model('swin_base_patch4_window12_384', num_classes = num_classes, pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    #images divided in squared patches of 4x4 pixels
    #attention window size 7x7 meaning that every token(patch) could be influenced by max 7 neighbors tokens
    #image size required in input 224x224
    model3 = timm.create_model('swin_base_patch4_window7_224', num_classes = num_classes, pretrained=True)

    #Not working
    model4 = timm.create_model('swin_large_patch4_window12_384', num_classes = num_classes, pretrained=True)
    model.to(DEVICE)
    if wandb_active:
        wandb.login()
        wandb.init(project="TeamBananaBread")
        wandb.watch(model)

    # Now comes the training
    loss_fn = nn.CrossEntropyLoss()
    #optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-8)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-7)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-7)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer=optimizer, scheduler=scheduler, log=wandb_active)
        competition_test_loop(test_loader, model, train_loader.dataset.classes)
    print("Done!")


if __name__ == '__main__':
    main()


#CosineAnnealingWarmRestarts(optimizer, T_0=30, eta_min=1e-2, last_epoch=-1)
#optim.NAdam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
#NB increasing weight_decay lead to reduction of overfitting
#1h40min NAdam base 384 not better results compared to AdamW
#19:40