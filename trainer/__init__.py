import torch
import wandb

from utils import get_project_root

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = get_project_root() / "results"


def train_loop(dataloader, model, loss_fn, optimizer, scheduler=None, num_classifiers=1, log=False, save_name=None):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        total_loss = 0
        correct = 0
        # Compute prediction and loss
        if num_classifiers == 1:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        else:
            for i in range(num_classifiers):
                pred = model(X)[i]
                loss = loss_fn(pred, y)
                if i==num_classifiers-1:
                    loss = loss * 2
                if log:
                    wandb.log({f"train/cl{i}_loss": loss.item()})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                if i == num_classifiers - 1:  # the last classifier is the main classifier, only take accuracy here
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= len(y)
        if batch % 20 == 0:
            loss, current = total_loss, batch * dataloader.batch_size + len(X)
            print(f"loss: {total_loss:>7f}, acc: {(100 * correct):>0.1f}%,  [{current:>5d}/{size:>5d}]")

        if log:
            wandb.log({"train/loss": total_loss, "train/acc": correct})

    if scheduler:
        scheduler.step()

    if save_name:
        torch.save(model.state_dict(), RESULTS_DIR / save_name)


# this is copied from pytorch tutorial, seems general enough
def test_loop(dataloader, model, loss_fn, log=False):
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
            pred = model(X)
            if isinstance(pred, tuple):
                pred = pred[-1]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if log:
        wandb.log({"val/accuracy": correct, "val/loss": test_loss})
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
