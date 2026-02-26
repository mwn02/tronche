import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from network.main.network import Network

# transform the data to a tensor
transform = torchvision.transforms.Compose([
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	transforms.Grayscale(num_output_channels=1),
])

# Load the training and test data
training_data = torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=transform)
train_data, test_data = random_split(training_data, [0.8, 0.2])

batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Define the device and the model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = Network(device)

# Define the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 5
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	model.train_model(train_dataloader, loss_fn, optimizer)
	model.test_model(test_dataloader, loss_fn)
print("Done!")

torch.save(model.state_dict(), "network/main/model.pth")
print("Saved PyTorch Model State to model.pth")

