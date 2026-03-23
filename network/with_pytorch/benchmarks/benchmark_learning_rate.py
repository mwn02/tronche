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
learning_rates_to_test = [0.1, 0.05, 0.01]

plt.figure(figsize=(10, 6))

for lr in learning_rates_to_test:
    print(f"\n=== Learning rate = {lr} ===")

    torch.manual_seed(42)
    model = Network(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    accuracies = []
    for t in range(epochs):
        model.train_model(train_dataloader, loss_fn, optimizer)
        _, accuracy = model.test_model(test_dataloader, loss_fn)
        accuracies.append(accuracy)
        print(f"  Époque {t+1} - Accuracy: {accuracy:.1f}%")

    plt.plot(range(1, epochs + 1), accuracies, label=f"LR = {lr}")

plt.title("Précision selon le learning rate")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("network/with_pytorch/comparaison_learning_rate.png")
plt.show()
print("\nGraphique sauvegardé!")
