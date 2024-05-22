import torch
import torch.nn as nn
import torch.nn.functional as F


class SubsetDFCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SubsetDFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example feature subsets
subset1 = torch.arange(0, 33)
subset2 = torch.arange(33, 66)
subset3 = torch.arange(66, 100)

num_classes = 10  # Example for 10-class classification

# Instantiate models for each subset
models = [SubsetDFCNN(len(subset1), num_classes),
          SubsetDFCNN(len(subset2), num_classes),
          SubsetDFCNN(len(subset3), num_classes)]

# Define loss function and optimizer for each model
criterions = [nn.CrossEntropyLoss() for _ in models]
optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in models]

# Example training loop for each model
num_epochs = 10
for epoch in range(num_epochs):
    for i, model in enumerate(models):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:  # Assuming you have a DataLoader named `train_loader`
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Use only the subset of features for this model
            subset_inputs = inputs[:, subset1] if i == 0 else inputs[:, subset2] if i == 1 else inputs[:, subset3]

            optimizers[i].zero_grad()

            outputs = model(subset_inputs)
            loss = criterions[i](outputs, labels)

            loss.backward()
            optimizers[i].step()

            running_loss += loss.item()

        print(f"Model {i + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Training complete!")


def ensemble_predict(models, inputs):
    outputs = []
    for i, model in enumerate(models):
        subset_inputs = inputs[:, subset1] if i == 0 else inputs[:, subset2] if i == 1 else inputs[:, subset3]
        outputs.append(model(subset_inputs))

    # Stack outputs and apply max voting
    outputs = torch.stack(outputs, dim=0)
    _, final_predictions = torch.max(outputs.sum(dim=0), dim=1)
    return final_predictions


# Example inference
ensemble_model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:  # Assuming you have a DataLoader named `test_loader`
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        predictions = ensemble_predict(models, inputs)
        # Evaluate predictions
