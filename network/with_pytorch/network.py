from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, device: str, hidden_size: int = 512, num_conv_layers: int = 1, num_dense_layers: int = 2, activation: str = 'relu'):
        super().__init__()
        self.device = device

        if num_conv_layers < 1 or num_conv_layers > 3:
            raise ValueError(f"num_conv_layers must be between 1 and 3, got {num_conv_layers}")
        if activation not in ('relu', 'sigmoid'):
            raise ValueError(f"activation must be 'relu' or 'sigmoid', got {activation}")

        # Build conv stack
        conv_channels = [1, 32, 64, 128]
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers += [
                nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_size=3, stride=1, padding=1),
                nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        self.convolutional_layer = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        
        self.linear_layer = nn.Sequential(
            nn.Linear(32*14*14, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
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
