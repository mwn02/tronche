import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from network.with_pytorch.network import Network

# Setup des données

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

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
epochs = 50
learning_rates_to_test = [0.1, 0.05, 0.01]

# Préparation du graphique
plt.figure(figsize=(10, 6))

# Boucle de comparaison
for lr in learning_rates_to_test:
    print(f"\n=== Test avec Learning Rate = {lr} ===")
    
    # Réinitialiser le modèle pour chaque test
    model = Network(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    accuracies_for_this_lr = []
    
    # Entraînement pour N époques
    for t in range(epochs):
        # On entraîne
        model.train_model(train_dataloader, loss_fn, optimizer)
        
        # Test et on récupération de l'accuracy
        avg_loss, accuracy = model.test_model(test_dataloader, loss_fn)
        
        # Sauvegarde de l'accuracy pour cette époque
        accuracies_for_this_lr.append(accuracy)
        print(f"Époque {t+1} - Accuracy: {accuracy:.1f}%")
        
    # Ajout de la ligne au graphique
    plt.plot(range(1, epochs + 1), accuracies_for_this_lr, label=f"LR = {lr}")

# Afficher et sauvegarder le graphique
plt.title("Évolution de la précision en fonction des époques selon le taux d'apprentissage")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("network/with_pytorch/comparaison_lr_.png")
plt.show()
print("\nGraphique sauvegardé!")
