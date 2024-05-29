"""
"""
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights


class WSS(nn.Module):
    def __init__(self, in_channel: int, num_classes: int, num_selects: int):
        super().__init__()
        self.fc = nn.Linear(in_channel, num_classes)
        self.num_selects = num_selects

    def forward(self, x):
        # x = [B X H X W X F] for input
        x = torch.flatten(x, start_dim=1, end_dim=2)  # combine HXW
        # c = [B X (HXW) X C]
        logits = torch.softmax(self.fc(x), dim=-1)
        scores, _ = torch.max(logits, dim=-1)
        _, ids = torch.sort(scores, dim=-1, descending=True)
        selection = ids[:, :self.num_selects]
        return logits, torch.gather(x, 1, selection.unsqueeze(-1))


class HERBS(nn.Module):
    transforms = Swin_V2_T_Weights.DEFAULT.transforms

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        for parameter in self.backbone.parameters(recurse=True):
            parameter.requires_grad = False

        # for swin the odd layers are the feature extractors
        return_nodes = {
            f"features.{k}": f"extractor_{i}" for i, k in enumerate([1, 3, 5, 7])
        }
        self.features = create_feature_extractor(self.backbone, return_nodes)

        num_selects = [256, 128, 64, 32]  # the number of features we select at each stage
        self.suppressors = nn.ModuleDict()
        for i, in_channel in enumerate(self.get_feature_dims()):
            self.suppressors[f"extractor_{i}"] = WSS(in_channel, num_classes, num_selects[i])

        self.classifier = nn.Linear(sum(num_selects), num_classes)

    def forward(self, x):
        features = self.features(x)
        out = [self.suppressors[node](feature_map) for node, feature_map in features.items()]
        selected_features = torch.cat([o[1].squeeze(-1) for o in out], dim=-1)
        return self.classifier(selected_features)

    def get_feature_dims(self):
        inp = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.features(inp)
        in_channels_list = [o.shape[-1] for o in out.values()]
        return in_channels_list
