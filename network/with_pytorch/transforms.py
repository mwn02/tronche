import torch
from PIL import Image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

def crop_black(image: Image.Image):
    """
    Given a PIL Image, it crops the unecessary white space around the image, and returns the image.
    """
    gray = np.array(image.convert("L")) / 255.0 # convert to grayscale (0 is black, 1 is white)
    mask = gray < 0.15 # mask for dark values

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0) # computes the min per column -> gives the top-left corner
    y_max, x_max = coords.max(axis=0) # computes the max per column -> gives the bottom-right corner

    return image.crop((x_min, y_min, x_max + 1, y_max + 1)) # add +1 because the boundaries are excluded


def preview_transformation(path):
    """
    given the image's path, tests out the transformation and shows the output image
    """
    img = Image.open(path)
    transform = v2.Compose([
        v2.Lambda(crop_black),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((32, 32), antialias=False),
        v2.Grayscale(num_output_channels=1),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=255),
        v2.GaussianNoise(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    t_img = transform(img)
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.show()
    plt.imshow(t_img.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.title("Plot of torchvision.tv_tensors.Image")
    plt.axis('off') # Optional: hide axes ticks
    plt.show()

def get_base_transform():
    return v2.Compose([
        v2.Lambda(crop_black),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((32, 32), antialias=False),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
    ])

def get_train_transform(mean, std):
    return v2.Compose([
        v2.Lambda(crop_black),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((32, 32), antialias=False),
        v2.Grayscale(num_output_channels=1),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=255),
        v2.GaussianNoise(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

def get_test_transform(mean, std):
    return v2.Compose([
        v2.Lambda(crop_black),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((32, 32), antialias=False),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

if __name__ == "__main__":
    pass