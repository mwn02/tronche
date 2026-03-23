import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network.with_pytorch.network import Network
from network.with_pytorch.data_fetching import get_emoji_data

train_data, test_data = get_emoji_data(42)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- Benchmark ---
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
epochs = 50
activations_to_test = ['relu', 'sigmoid']

plt.figure(figsize=(10, 6))

for activation in activations_to_test:
    print(f"\n=== Activation = {activation} ===")

    torch.manual_seed(42)
    model = Network(device, activation=activation)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    accuracies = []
    for t in range(epochs):
        model.train_model(train_dataloader, loss_fn, optimizer)
        _, accuracy = model.test_model(test_dataloader, loss_fn)
        accuracies.append(accuracy)
        print(f"  Époque {t+1} - Accuracy: {accuracy:.1f}%")

    plt.plot(range(1, epochs + 1), accuracies, label=activation.upper())

plt.title("Précision selon la fonction d'activation (ReLU vs Sigmoid)")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("network/with_pytorch/comparaison_activation.png")
plt.show()
print("\nGraphique sauvegardé!")
