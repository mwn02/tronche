import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']
from torch.utils.data import DataLoader
from network.with_pytorch.network import Network
from network.with_pytorch.data_fetching import get_emoji_data

EMOJIS = ["🙂", "☹️", "❤️", "😭", "🤓"]

train_data, test_data = get_emoji_data(42)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
epochs = 30

torch.manual_seed(42)
model = Network(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for t in range(epochs):
    model.train_model(train_dataloader, loss_fn, optimizer)
    _, accuracy = model.test_model(test_dataloader, loss_fn)
    print(f"Époque {t+1} - Accuracy: {accuracy:.1f}%")

# --- Précision par classe ---
correct = [0] * 5
total = [0] * 5

model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        for label, pred in zip(y, preds):
            total[label.item()] += 1
            if pred.item() == label.item():
                correct[label.item()] += 1

accuracies = [100 * correct[i] / total[i] for i in range(5)]

# --- Graphique ---
plt.figure(figsize=(8, 5))
bars = plt.bar(EMOJIS, accuracies, color="crimson")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=11)
plt.title("Figure 4 : Précision par classe d'emoji")
plt.ylabel("Précision (%)")
plt.ylim(0, 110)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("network/with_pytorch/comparaison_precision_par_classe.png")
plt.show()
print("\nGraphique sauvegardé!")
