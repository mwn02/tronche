import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# --- 1. Load MNIST dataset ---
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# --- 2. Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# --- 3. Define the model ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # output logits for 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits  # raw logits

model = NeuralNetwork().to(device)
print(model)

# --- 4. Loss and optimizer ---
loss_fn = nn.BCEWithLogitsLoss()  # binary cross entropy with logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training function ---
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Convert labels to one-hot floats
        y_onehot = F.one_hot(y, num_classes=10).float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y_onehot)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# --- 6. Test function ---
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_onehot = F.one_hot(y, num_classes=10).float()  # convert for BCE
            pred = model(X)
            test_loss += loss_fn(pred, y_onehot).item()

            # Convert logits to probabilities for accuracy
            probs = torch.sigmoid(pred)  # BCE predicts independent probabilities
            predicted_class = probs.argmax(dim=1)
            correct += (predicted_class == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# --- 7. Training loop ---
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# --- 8. Save the model ---
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
