import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from network.with_pytorch.data_fetching import get_emoji_data

# ---------- SHOW A FEW INDIVIDUAL IMAGES ----------
def show_samples(subset, n=5):
    for i in range(n):
        img, label = subset[i]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)  # C,H,W → H,W,C

        plt.imshow(img, cmap='gray' if img.ndim == 2 or img.shape[-1] == 1 else None)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


# ---------- SHOW A GRID (BETTER) ----------
def show_batch(subset, batch_size=8):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))

    images = images.detach().cpu()

    grid = vutils.make_grid(images, nrow=4)
    grid = grid.permute(1, 2, 0)  # C,H,W → H,W,C

    plt.imshow(grid)
    plt.axis("off")
    plt.show()

def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def show_batch_with_labels(subset, batch_size=8):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))

    images = images.detach().cpu()
    labels = labels.detach().cpu()

    cols = 4
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]

        if i >= batch_size:
            ax.axis("off")
            continue

        img = images[i]

        # C,H,W → H,W,C
        if img.ndim == 3:
            img = img.permute(1, 2, 0)

        # grayscale handling
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap="gray")
        else:
            ax.imshow(img)

        ax.set_title(f"{labels[i].item()}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    train_data, _ = get_emoji_data()
    show_batch_with_labels(train_data, batch_size=32)