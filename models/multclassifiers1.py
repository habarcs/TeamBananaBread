from torch import nn, cat
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size, stride=1, padding=padding)
        self.batch_normalization = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_normalization(self.conv(x)))


class FeatureClassifier(nn.Module):
    def __init__(self, max_pool_kernel, in_channel, out_channel, num_classes):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv2 = ConvolutionBlock(out_channel, 2 * out_channel, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(max_pool_kernel, stride=1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2 * out_channel),
            nn.Linear(2 * out_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ELU(),
            nn.Linear(out_channel, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        feature = x.view(x.size(0), -1)
        pred_class = self.classifier(feature)
        return pred_class, feature


class MultiClassifier(nn.Module):
    transforms = ResNet50_Weights.DEFAULT.transforms

    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters(True):
            param.requires_grad = False
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        self.features = create_feature_extractor(self.resnet, return_nodes)
        parameters = {
            'layer1': {
                "max_pool_kernel": 112,
                "in_channel": 256,
                "out_channel": 512,
                "num_classes": num_classes
            },
            'layer2': {
                "max_pool_kernel": 56,
                "in_channel": 512,
                "out_channel": 512,
                "num_classes": num_classes
            },
            'layer3': {
                "max_pool_kernel": 28,
                "in_channel": 1024,
                "out_channel": 512,
                "num_classes": num_classes
            },
            'layer4': {
                "max_pool_kernel": 14,
                "in_channel": 2048,
                "out_channel": 512,
                "num_classes": num_classes
            },
        }
        self.classifiers = nn.ModuleDict()
        for node in return_nodes:
            self.classifiers[node] = FeatureClassifier(**parameters[node])
            for param in self.classifiers[node].max_pool.parameters():
                param.requires_grad = False
            

        self.concat_classifier = nn.Sequential(
            nn.BatchNorm1d(1024 * 4),
            nn.Linear(1024 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        extracted_maps = self.features(x)
        class_pred_out = []
        feature_out = []
        for node, feature_map in extracted_maps.items():
            pred, feature = self.classifiers[node](feature_map)
            class_pred_out.append(pred)
            feature_out.append(feature)

        cat_out = self.concat_classifier(cat(feature_out, -1))

        return *class_pred_out, cat_out
