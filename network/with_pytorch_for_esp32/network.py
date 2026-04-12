from torch import nn
import torch


class Network(nn.Module):
    # Reduced default hidden_size for ESP32 constraints (was 512, 512)
    def __init__(
        self,
        device: str = "cpu",
        activation: str = "relu",
        hidden_size: list[int] = [64, 32],
    ):
        super().__init__()
        self.device = device

        if activation not in ("relu", "sigmoid"):
            raise ValueError(
                f"activation must be 'relu' or 'sigmoid', got {activation}"
            )

        # Adding an extra Conv + Pool layer drastically reduces the spatial
        # dimensions before the Linear layer, which saves megabytes of RAM.
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )

        self.flatten = nn.Flatten()

        # 16 channels * 8 width * 8 height = 1024 parameters flattened
        self.linear_layer = nn.Sequential(
            nn.Linear(16 * 8 * 8, hidden_size[0]),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Linear(hidden_size[1], 5),
        )
        self.to(device)

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x

    def train_model(self, dataloader, loss_fn, optimizer):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def test_model(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        avg_loss = test_loss / num_batches
        accuracy = 100 * (correct / size)

        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f}")

        return avg_loss, accuracy
