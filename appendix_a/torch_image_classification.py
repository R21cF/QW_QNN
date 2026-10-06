# Image classification on Fashion-MNIST with a fully connected network.
# Diagnostic prints, model saving/reloading and
# a single-image demonstration are omitted; 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transform = v2.Compose([v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)])
training_data = datasets.FashionMNIST(root="data", train=True,
                                      download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=True, transform=transform)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else "cpu")

# The model: two hidden layers of 512 rectified-linear units.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.stack(self.flatten(x))

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# One pass over the training set: forward, loss, backward, step.
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluation on the held-out set.
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    n, nb = len(dataloader.dataset), len(dataloader)
    print(f"accuracy {100 * correct / n:.1f}%, avg loss {test_loss / nb:.6f}")

for epoch in range(5):
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
