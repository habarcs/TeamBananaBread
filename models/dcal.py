"""
Hello World, my first ML program
This model is inspired by the following paper:
https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Dual_Cross-Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Object_Re-Identification_CVPR_2022_paper.pdf
Some parts related to text and other context are omitted from the model
- usual models use spatial attention, or localizing the objects or parts
- this model uses self-attention instead
"""
import torch
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision.datasets.fgvc_aircraft import FGVCAircraft
from torch.utils.data.dataloader import DataLoader
from torch import nn
from utils import get_project_root

model = vit_b_16(representation_size=100)  # we force it to be 100 classes because we are using the airplane data set
# now we do data loading part
data_set_train = FGVCAircraft(root=get_project_root() / "data",
                              split="train",
                              annotation_level="variant",
                              transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
                              target_transform=None,
                              download=True)
data_set_test = FGVCAircraft(root=get_project_root() / "data",
                             split="test",
                             annotation_level="variant",
                             transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
                             target_transform=None,
                             download=True)

data_loader_train = DataLoader(batch_size=69, shuffle=True, dataset=data_set_train)

data_loader_test = DataLoader(batch_size=69, shuffle=True, dataset=data_set_test)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=1e-4)

for batch, (X, y) in enumerate(data_loader_train):
    size = len(data_loader_train.dataset)
    model.train()
    pred = model(X) # we get the prediction by giving the model X (our images)
    loss = loss_fn(pred, y) # we get loss by using loss fn giving it the label prediction and the original label
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 10 == 0:
        loss, current = loss.item(), batch * data_loader_train.batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
