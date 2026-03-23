import numpy as np
import torchvision
from torch.utils.data import DataLoader, Subset
from network.with_pytorch.transforms import get_base_transform, get_train_transform, get_test_transform

def get_emoji_data(seed=None):
    # Load and Split (Ensures different transforms for train/test)
    full_dataset = torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=get_base_transform())
    indices = np.arange(len(full_dataset))
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]

    # compute mean and std
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    mean = 0.
    std = 0.
    total_pixels = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # change shape (B, C, H, W) -> (B, C, H*W)

        mean += images.sum(2).sum(0) # sum over each pixel, then over each sum of pixels
        std += (images ** 2).sum(2).sum(0) # square over each pixel
        total_pixels += images.size(0) * images.size(2) # batch size X number of pixels => total pixels

    mean /= total_pixels # average over all pixels
    std = (std / total_pixels - mean ** 2).sqrt()
    print(f"mean: {mean.tolist()}, std: {std.tolist()}")

    train_transform = get_train_transform(mean.tolist(), std.tolist())
    test_transform = get_test_transform(mean.tolist(), std.tolist())
    train_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=train_transform), train_idx)
    test_data = Subset(torchvision.datasets.ImageFolder(root="dataset/dataset-data/training-data", transform=test_transform), test_idx)
    return train_data, test_data

if __name__ == "__main__":
    pass