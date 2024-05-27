import torch

from utils import get_project_root

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = get_project_root() / "results"


def train_loop(dataloader, model, loss_fn, optimizer, scheduler=None, num_classifiers=1):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction and loss
        if num_classifiers == 1:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
        else:  # handles multiple classifiers, by doing a gradient descent on each of them
            preds = model(X)[:num_classifiers]
            loss = 0
            for pred in reversed(preds):
                optimizer.zero_grad()
                single_loss = loss_fn(pred, y) * 1
                single_loss.backward()
                optimizer.step()
                loss += single_loss.item()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    if scheduler:
        scheduler.step()


# this is copied from pytorch tutorial, seems general enough
def test_loop(dataloader, model, loss_fn, num_classifiers=1):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            if num_classifiers == 1:
                pred = model(X)
            else:
                pred = model(X)[num_classifiers]  # the last loss is the loss of the concat classifier
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
