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
hidden_sizes_to_test = [128, 256, 512, 1024]

plt.figure(figsize=(10, 6))

for hidden_size in hidden_sizes_to_test:
    print(f"\n=== Hidden size = {hidden_size} ===")

    torch.manual_seed(42)
    model = Network(device, hidden_size=hidden_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    accuracies = []
    for t in range(epochs):
        model.train_model(train_dataloader, loss_fn, optimizer)
        _, accuracy = model.test_model(test_dataloader, loss_fn)
        accuracies.append(accuracy)
        print(f"  Époque {t+1} - Accuracy: {accuracy:.1f}%")

    plt.plot(range(1, epochs + 1), accuracies, label=f"Hidden size = {hidden_size}")

plt.title("Précision selon la taille des couches denses")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("network/with_pytorch/comparaison_layer_size.png")
plt.show()
print("\nGraphique sauvegardé!")
