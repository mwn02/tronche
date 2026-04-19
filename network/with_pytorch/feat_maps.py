import torch
import matplotlib.pyplot as plt
from math import ceil, sqrt
from network.with_pytorch.network import Network
from network.with_pytorch.data_fetching import get_emoji_data


def save_single_image(tensor, filename, title="Input image"):
    """
    tensor: shape (1, H, W) or (H, W)
    """
    img = tensor.detach().cpu().squeeze()

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_maps_grid(feature_maps, filename, title="", max_maps=None):
    """
    feature_maps: tensor of shape (1, C, H, W)
    Saves all or first max_maps channels in a clean grid.
    """
    fm = feature_maps.detach().cpu()

    if fm.dim() != 4 or fm.shape[0] != 1:
        raise ValueError(f"Expected feature maps of shape (1, C, H, W), got {tuple(fm.shape)}")

    num_channels = fm.shape[1]
    if max_maps is not None:
        num_channels = min(num_channels, max_maps)

    cols = int(ceil(sqrt(num_channels)))
    rows = int(ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(num_channels):
        axes[i].imshow(fm[0, i], cmap="gray")
        axes[i].set_title(f"Map {i}")
        axes[i].axis("off")

    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_feature_maps(model, image_tensor, output_prefix="feature_maps"):
    """
    image_tensor: shape (1, H, W)
    Saves:
      - original image
      - conv output
      - relu output
      - pool output
    """
    activations = {}

    def conv_hook(module, inp, out):
        activations["conv"] = out.detach()

    def relu_hook(module, inp, out):
        activations["relu"] = out.detach()

    def pool_hook(module, inp, out):
        activations["pool"] = out.detach()

    # Hooks on your sequential layer:
    # [0] Conv2d
    # [1] ReLU
    # [2] MaxPool2d
    h1 = model.convolutional_layer[0].register_forward_hook(conv_hook)
    h2 = model.convolutional_layer[1].register_forward_hook(relu_hook)
    h3 = model.convolutional_layer[2].register_forward_hook(pool_hook)

    model.eval()
    device = model.device

    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)
        _ = model(x)

    h1.remove()
    h2.remove()
    h3.remove()

    # Save original image
    save_single_image(image_tensor, f"{output_prefix}_input.png", title="Input image")

    # Save feature maps
    save_feature_maps_grid(
        activations["conv"],
        f"{output_prefix}_conv.png",
        title="Feature maps after Conv2d"
    )

    save_feature_maps_grid(
        activations["relu"],
        f"{output_prefix}_relu.png",
        title="Feature maps after ReLU"
    )

    save_feature_maps_grid(
        activations["pool"],
        f"{output_prefix}_pool.png",
        title="Feature maps after MaxPool2d"
    )

    print("Saved:")
    print(f"  {output_prefix}_input.png")
    print(f"  {output_prefix}_conv.png")
    print(f"  {output_prefix}_relu.png")
    print(f"  {output_prefix}_pool.png")


if __name__ == "__main__":
    # Load data
    train_data, test_data = get_emoji_data()

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Load model
    model = Network(device)
    model.load_state_dict(torch.load("network/saved_models/model_v1.pth", map_location=device))

    # Pick one image
    image, label = test_data[300]   # image shape should be (1, 32, 32)

    # Create visualizations
    visualize_feature_maps(model, image, output_prefix="example_feature_maps")