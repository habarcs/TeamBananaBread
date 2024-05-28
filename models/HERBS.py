"""
"""
from torch import nn
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights


class HERBS(nn.Module):
    transforms = Swin_V2_T_Weights.DEFAULT.transforms

    def __init__(self, number_of_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        for parameter in self.backbone.parameters(recurse=True):
            parameter.requires_grad = False
        self.backbone.head = nn.Linear(1024, number_of_classes)

    def forward(self, x):
        return self.backbone(x)
