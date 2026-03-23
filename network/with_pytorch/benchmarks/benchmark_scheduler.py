import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from network.with_pytorch.network import Network
from network.with_pytorch.data_fetching import get_emoji_data

train_data, test_data = get_emoji_data(42)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- Benchmark ---
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
epochs = 50
configs = [
    {"label": "Pas de scheduler", "use_scheduler": False},
    {"label": "ReduceLROnPlateau", "use_scheduler": True},
]

plt.figure(figsize=(10, 6))

for config in configs:
    print(f"\n=== {config['label']} ===")

    torch.manual_seed(42)
    model = Network(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5) if config["use_scheduler"] else None

    accuracies = []
    for t in range(epochs):
        model.train_model(train_dataloader, loss_fn, optimizer)
        avg_loss, accuracy = model.test_model(test_dataloader, loss_fn)
        if scheduler is not None:
            scheduler.step(avg_loss)
        accuracies.append(accuracy)
        print(f"  Époque {t+1} - Accuracy: {accuracy:.1f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

    plt.plot(range(1, epochs + 1), accuracies, label=config["label"])

plt.title("Précision avec et sans learning rate scheduler")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("network/with_pytorch/comparaison_scheduler.png")
plt.show()
print("\nGraphique sauvegardé!")
