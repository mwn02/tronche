from typing import Any
from torch import nn
import torch


class Network(nn.Module):
	def __init__(self, device: str):
		super().__init__()
		self.device = device
		
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
				nn.Linear(32*32, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 5)
			)
		self.softmax = nn.Softmax(dim=1)
		self.to(device)
	
	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return self.softmax(logits)

	def train_model(self, dataloader, loss_fn, optimizer):
		size = len(dataloader.dataset)
		self.train()

		for batch, (X, y) in enumerate[Any](dataloader):
			X, y = X.to(self.device), y.to(self.device)
			pred = self.forward(X)
			loss = loss_fn(pred, y)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			if batch % 100 == 0:
				loss, current = loss.item(), (batch + 1) * len(X)
				print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	def test_model(self, dataloader, loss_fn):
		size = len(dataloader.dataset)
		self.eval() # sets the model in evaluation mode
		test_loss, correct = 0, 0

		with torch.no_grad():
			for X, y in dataloader:
				X, y = X.to(self.device), y.to(self.device)
				pred = self(X)
				test_loss += loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		test_loss /= size
		correct /= size
		print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")