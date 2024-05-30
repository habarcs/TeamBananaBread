"""
"""
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights
from torchvision.ops import MLP


class WSS(nn.Module):
    def __init__(self, in_embedding_size: int, num_classes: int, num_selects: int, out_embedding_size: int):
        super().__init__()
        self.fc = nn.Linear(in_embedding_size, num_classes)
        self.num_selects = num_selects
        self.in_embedding_size = in_embedding_size
        self.out_embedding_size = out_embedding_size

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)  # combine HXW
        pred = self.fc(x)
        logits = torch.softmax(pred, dim=-1)
        scores, _ = torch.max(logits, dim=-1)
        _, ids = torch.sort(scores, dim=-1, descending=True)
        indexes = ids[:, :self.num_selects]
        expanded_indexes = indexes.unsqueeze(-1).expand(-1, -1, self.in_embedding_size)
        selected_features = x.gather(dim=1, index=expanded_indexes)
        padded_selected = torch.nn.functional.pad(selected_features,
                                                  (0, self.out_embedding_size - self.in_embedding_size), "constant", 0)
        return torch.mean(pred, dim=1), padded_selected


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

        num_selects = [32, 32, 32, 32]  # the number of features we select at each stage
        self.suppressors = nn.ModuleDict()
        embedding_sizes = self.get_dims()
        for i in range(4):
            self.suppressors[f"extractor_{i}"] = WSS(embedding_sizes[i], num_classes, num_selects[i],
                                                     embedding_sizes[-1])

        input_layer = sum(num_selects) * embedding_sizes[-1]
        hidden_layer = int((input_layer + num_classes) * 2 / 3)
        self.classifier = MLP(input_layer, [hidden_layer, num_classes], dropout=0.3)

    def forward(self, x):
        features = self.features(x)
        out = [self.suppressors[node](feature_map) for node, feature_map in features.items()]
        selected_features = torch.cat([o[1] for o in out], dim=1)
        suppressor_preds = [o[0] for o in out]
        return *suppressor_preds, self.classifier(torch.flatten(selected_features, start_dim=1))

    def get_dims(self):
        inp = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.features(inp)
        embedding_size = [o.shape[-1] for o in out.values()]
        return embedding_size
