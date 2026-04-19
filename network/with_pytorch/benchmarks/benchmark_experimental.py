import sys
import io
import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']

from network.with_pytorch.network import Network
from network.with_pytorch.data_fetching import get_emoji_data
from network.with_pytorch.transforms import get_base_transform, get_test_transform

EMOJIS = ["🙂", "☹️", "❤️", "😭", "🤓"]
EMOJI_TO_IDX = {emoji: i for i, emoji in enumerate(EMOJIS)}
COLLECTED_DIR = Path("collected")
MODEL_PATH = Path("network/saved_models/model_v1_seed42.pth")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# --- Données ---
_, test_data = get_emoji_data(42)

# --- Mean/std depuis le dataset complet (même méthode que data_fetching.py) ---
full_dataset = torchvision.datasets.ImageFolder(
    root="dataset/dataset-data/training-data",
    transform=get_base_transform()
)
stats_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
mean = 0.
std = 0.
total_pixels = 0
for images, _ in stats_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.sum(2).sum(0)
    std += (images ** 2).sum(2).sum(0)
    total_pixels += images.size(0) * images.size(2)
mean /= total_pixels
std = (std / total_pixels - mean ** 2).sqrt()
mean = mean.tolist()
std = std.tolist()

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- Charger le modèle ---
model = Network(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- Précision théorique (test set) ---
correct_th = [0] * len(EMOJIS)
total_th = [0] * len(EMOJIS)
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        for label, pred in zip(y, preds):
            total_th[label.item()] += 1
            if pred.item() == label.item():
                correct_th[label.item()] += 1
acc_th = [100 * correct_th[i] / total_th[i] for i in range(len(EMOJIS))]
acc_th_global = 100 * sum(correct_th) / sum(total_th)

# --- Précision expérimentale (collected/) ---
transform = get_test_transform(mean, std)
correct_exp = [0] * len(EMOJIS)
total_exp = [0] * len(EMOJIS)
with torch.no_grad():
    for emoji, idx in EMOJI_TO_IDX.items():
        folder = COLLECTED_DIR / emoji
        if not folder.exists():
            print(f"Warning: dossier manquant pour {emoji}")
            continue
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            pred = model(tensor).argmax(1).item()
            total_exp[idx] += 1
            if pred == idx:
                correct_exp[idx] += 1
acc_exp = [100 * correct_exp[i] / total_exp[i] if total_exp[i] > 0 else 0.0 for i in range(len(EMOJIS))]
acc_exp_global = 100 * sum(correct_exp) / sum(total_exp) if sum(total_exp) > 0 else 0.0

# --- Tableau console ---
print("\n" + "="*55)
print(f"{'Classe':<10} {'Théorique':>12} {'Expérimental':>14} {'Écart':>10}")
print("-"*55)
for i, emoji in enumerate(EMOJIS):
    ecart = acc_exp[i] - acc_th[i]
    print(f"{emoji:<10} {acc_th[i]:>11.1f}% {acc_exp[i]:>13.1f}% {ecart:>+10.1f}%")
print("-"*55)
ecart_global = acc_exp_global - acc_th_global
print(f"{'GLOBAL':<10} {acc_th_global:>11.1f}% {acc_exp_global:>13.1f}% {ecart_global:>+10.1f}%")
print("="*55 + "\n")

# --- Graphique ---
x = np.arange(len(EMOJIS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars_th = ax.bar(x - width/2, acc_th, width, label="Théorique", color="crimson", alpha=0.85)
bars_exp = ax.bar(x + width/2, acc_exp, width, label="Expérimental", color="steelblue", alpha=0.85)
for bar in bars_th:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in bars_exp:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(EMOJIS, fontsize=13)
ax.set_ylabel("Précision (%)")
ax.set_title("Précision théorique vs expérimentale par classe")
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
output_path = Path("network/with_pytorch/comparaison_theorique_vs_experimental.png")
plt.savefig(output_path)
plt.show()
print(f"Graphique sauvegardé : {output_path}")
