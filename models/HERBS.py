"""
"""
import torch
from torchvision.ops import Permute
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, ToImage, ToDtype, Normalize, InterpolationMode, \
    RandomCrop, RandomHorizontalFlip, GaussianBlur
from timm import create_model
from torch import nn


class WSS(nn.Module):
    def __init__(self, in_embedding_size: int, num_classes: int, num_selects: int, out_embedding_size: int):
        super().__init__()
        self.fc = nn.Linear(in_embedding_size, num_classes)
        self.num_selects = num_selects
        self.in_embedding_size = in_embedding_size
        self.out_embedding_size = out_embedding_size
        self.classify = nn.Sequential(
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )
        self.dimension_reduce = nn.AdaptiveAvgPool1d(out_embedding_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)  # combine HXW
        pred = self.fc(x)
        logits = torch.softmax(pred, dim=-1)
        scores, _ = torch.max(logits, dim=-1)
        _, ids = torch.sort(scores, dim=-1, descending=True)
        indexes = ids[:, :self.num_selects]
        expanded_indexes = indexes.unsqueeze(-1).expand(-1, -1, self.in_embedding_size)
        selected_features = x.gather(dim=1, index=expanded_indexes)
        return self.classify(pred), self.dimension_reduce(selected_features)


class HERBS(nn.Module):
    transforms_train = Compose([
        Resize(size=510, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        RandomCrop(size=(384, 384)),
        RandomHorizontalFlip(),
        GaussianBlur(3),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ])
    transforms_test = Compose([
        Resize(size=510, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        CenterCrop(size=(384, 384)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ])

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = create_model("swin_base_patch4_window12_384", pretrained=True, features_only=True)
        for param in self.backbone.parameters(True):
            param.requires_grad = False

        num_selects = [32, 32, 32, 32]  # the number of features we select at each stage
        self.suppressors = nn.ModuleList()
        embedding_sizes = self.get_dims()
        for i in range(4):
            self.suppressors.append(WSS(embedding_sizes[i], num_classes, num_selects[i], embedding_sizes[0]))

        input_layer = sum(num_selects) * embedding_sizes[0]
        self.classifier = nn.Linear(input_layer, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = [sup(feature) for sup, feature in zip(self.suppressors, features)]
        selected_features = torch.cat([o[1] for o in out], dim=1)
        suppressor_preds = [o[0] for o in out]
        return *suppressor_preds, self.classifier(torch.flatten(selected_features, start_dim=1))

    def get_dims(self):
        inp = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            out = self.backbone(inp)
        embedding_size = [o.shape[-1] for o in out]
        return embedding_size
