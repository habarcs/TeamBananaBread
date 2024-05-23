#Trying out Subset Feature, second try with a bit of chat gpt help

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

#The data_transforms dictionary contains two keys: 'train' and 'val', each associated with a transforms.Compose object.
# This object is a sequential container that applies a list of transformations to the data.

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), #This transformation crops a random portion of the image and resizes it to 224x224 pixels.
        transforms.RandomHorizontalFlip(), #This randomly flips the image horizontally with a probability of 0.5.
        transforms.ToTensor(), #Converts the image (a PIL Image or numpy.ndarray) into a PyTorch tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #Normalizes the tensor image with selected mean and SD
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# This section of the code below is responsible for loading the data, applying the transformations,
# and setting up the data loaders for training and validation

#TODO change root directory to data set
data_dir = 'path_to_data' #this line specifies the root directory where the dataset is stored.

#this dictionary comprehension creates a dataset for both the training and validation sets
# using the ImageFolder class from torchvision.datasets.
#TODO change the image folder to the one we want
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#os.path.join(data_dir, x): Constructs the path to the dataset directories, e.g., path_to_data/train and path_to_data/val.
#data_transforms[x]: Applies the appropriate transformations (train or val) to the dataset

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
#This dictionary comprehension creates data loaders for both the training and validation datasets
# num_workers=4: Number of subprocesses to use for data loading. More workers can speed up data loading but requires more CPU resources.
#TODO CAN THIS BE CHANGED TO GPU?

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#This dictionary comprehension computes the size (number of images) of both the training and validation datasets.

class_names = image_datasets['train'].classes
#Purpose: Extracts the class names from the training dataset.
#image_datasets['train'].classes: The ImageFolder class automatically assigns a list of class names based on the
# sub-directory names. For instance, if the training dataset has sub-directories named 'dog' and 'cat', classes will be ['cat', 'dog']


### Learning Model
#The SubsetFeatureLearningModel class is a neural network model designed for fine-grained category classification,
# leveraging the principles of subset feature learning. This model uses a pre-trained ResNet50 backbone and adds both a
# main classifier and several subset classifiers.
#TODO do we want to change to another pretrained model?

class SubsetFeatureLearningModel(nn.Module):
    #Inheritance from nn.Module: The model inherits from torch.nn.Module, which is the base class for all neural network modules in PyTorch.
    def __init__(self, num_classes, subset_classes):
        super(SubsetFeatureLearningModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Removing the original classifier

        # Main classifier
        self.fc_main = nn.Linear(num_ftrs, num_classes)

        # Subset classifiers
        self.fc_subsets = nn.ModuleList([nn.Linear(num_ftrs, sub_classes) for sub_classes in subset_classes])

    def forward(self, x):
        x = self.backbone(x)
        main_out = self.fc_main(x)
        subset_outs = [fc(x) for fc in self.fc_subsets]
        return main_out, subset_outs


# Define the number of classes and subsets
num_classes = len(class_names)
subset_classes = [50, 50, 50, 50]  # Example subsets

model = SubsetFeatureLearningModel(num_classes, subset_classes)

def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    main_out, subset_outs = model(inputs)
                    _, preds = torch.max(main_out, 1)
                    loss_main = criterion(main_out, labels)
                    loss_subset = sum(criterion(sub_out, labels) for sub_out in subset_outs)
                    loss = loss_main + loss_subset

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = train_model(model, criterion, optimizer, num_epochs=25)

def evaluate_model(model):
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            main_out, _ = model(inputs)
            _, preds = torch.max(main_out, 1)

        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes['val']
    print(f'Validation Accuracy: {acc:.4f}')

evaluate_model(model)
