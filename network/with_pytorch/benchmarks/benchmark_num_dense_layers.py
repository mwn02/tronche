import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from network.with_pytorch.network import Network
from network.with_pytorch.main import crop_black

# --- Data pipeline (compute mean/std once) ---
default_transform = torchvision.transforms.Compose([
    transforms.Lambda(crop_black),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=default_transform)
indices = np.arange(len(full_dataset))
np.random.seed(42)
np.random.shuffle(indices)
train_idx = indices[:int(0.8 * len(indices))]
test_idx = indices[int(0.8 * len(indices)):]

loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
mean, std, total_pixels = 0., 0., 0
for images, _ in loader:
    b = images.size(0)
    images = images.view(b, images.size(1), -1)
    mean += images.sum(2).sum(0)
    std += (images ** 2).sum(2).sum(0)
    total_pixels += b * images.size(2)
mean /= total_pixels
std = (std / total_pixels - mean ** 2).sqrt()

train_transform = torchvision.transforms.Compose([
    transforms.Lambda(crop_black),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
])
test_transform = torchvision.transforms.Compose([
    transforms.Lambda(crop_black),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
])

train_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=train_transform), train_idx)
test_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=test_transform), test_idx)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- Benchmark ---
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
epochs = 50
num_dense_layers_to_test = [1, 2, 3]

plt.figure(figsize=(10, 6))

for num_dense in num_dense_layers_to_test:
    print(f"\n=== Couches denses cachées = {num_dense} ===")

    torch.manual_seed(42)
    model = Network(device, num_dense_layers=num_dense)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    accuracies = []
    for t in range(epochs):
        model.train_model(train_dataloader, loss_fn, optimizer)
        _, accuracy = model.test_model(test_dataloader, loss_fn)
        accuracies.append(accuracy)
        print(f"  Époque {t+1} - Accuracy: {accuracy:.1f}%")

    plt.plot(range(1, epochs + 1), accuracies, label=f"{num_dense} couche(s) dense cachée(s)")

plt.title("Précision selon le nombre de couches denses cachées")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("network/with_pytorch/comparaison_num_dense_layers.png")
plt.show()
print("\nGraphique sauvegardé!")
