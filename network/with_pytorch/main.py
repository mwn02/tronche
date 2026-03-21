import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from network.with_pytorch.network import Network
# 1. Training transforms: Includes Augmentation
train_transform = torchvision.transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 2. Testing transforms: Clean images for honest accuracy
test_transform = torchvision.transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 3. Load and Split (Ensures different transforms for train/test)
full_dataset = torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data")
indices = np.arange(len(full_dataset))
np.random.shuffle(indices)
train_idx = indices[:int(0.8 * len(indices))]
test_idx = indices[int(0.8 * len(indices)):]

train_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=train_transform), train_idx)
test_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=test_transform), test_idx)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# 4. Model Setup
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = Network(device)

loss_fn = torch.nn.CrossEntropyLoss()
# Using SGD with momentum as requested
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 5. The Scheduler (Watches test_loss to adjust LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# 6. Training Loop
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train_model(train_dataloader, loss_fn, optimizer)
    
    # This now captures the average loss returned by the network
    current_test_loss, _ = model.test_model(test_dataloader, loss_fn)
    
    # Update the learning rate based on performance
    scheduler.step(current_test_loss)
    
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")

print("Done!")
torch.save(model.state_dict(), "network/saved_models/model.pth")