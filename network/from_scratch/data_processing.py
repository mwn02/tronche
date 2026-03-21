import numpy as np
from PIL import Image, ImageOps
import random
import os
from torchvision import datasets, transforms

def get_emoji_data(base_path):
    """
    Retourne les listes de données d'entrainement X, et la liste d'étiquer y.
    
    La valeur retournée est X, y
    """

    all_images = []
    all_labels = []

    # 1. Itérer à travers chaque dossier (0, 1, 2, 3, 4)
    for label_folder in os.listdir(base_path):
        category_path = os.path.join(base_path, label_folder)
        
        if not os.path.isdir(category_path):
            continue

        # 2. Itérer à travers chaque image du dossier
        for image_name in os.listdir(category_path):
            full_path = os.path.join(category_path, image_name)
            
            img = Image.open(full_path)
            img = img.resize((32, 32), resample=Image.BILINEAR)
            img = ImageOps.grayscale(img)

            # Aumentation de données
            angle = random.uniform(-15, 15)
            tx = random.uniform(-0.1, 0.1) * 32
            ty = random.uniform(-0.1, 0.1) * 32

            img = img.rotate(
                angle, 
                resample=Image.BILINEAR, 
                expand=False, 
                translate=(tx, ty),
                fillcolor=0
            )

            # 3. NumPy Processing
            img_np = np.array(img).astype(np.float32) / 255.0
                
            # Normalizing to range [-1, 1]
            img_np = (img_np - 0.5) / 0.5
                
            # on ajoute une dimension pour que la forme soit (32, 32, 1)
            img_np = np.expand_dims(img_np, axis=-1) 
                
            all_images.append(img_np)
            all_labels.append(int(label_folder)) # Présumant que les dossiers sont 0, 1, 2...

    # 4. Conversion des listes en NumPy arrays, puis mélange
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y

def get_shuffled_data(X, y, num_classes, split_ratio):
    """
    Retroune les données mélangées d'entrainement et de test.

    La valeur retournée est: X_train, X_test, y_train, y_test
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split = int(split_ratio * len(X))

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    # One-hot encoding pour bien marcher avec softmax et cross-entropy
    y_train_hot = np.eye(num_classes)[y_train]
    y_test_hot = np.eye(num_classes)[y_test]

    return X_train, X_test, y_train_hot, y_test_hot

def load_mnist_data():
    """
    Retourne les données d'entrainement et de test (ainsi que leur étiquette) du MNIST dataset.

    La valeur retournée est: X_train, X_test, y_train, y_test
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    num_classes = 10

    def one_hot(y, num_classes=10):
        oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
        oh[np.arange(y.shape[0]), y] = 1.0
        return oh

    y_train = one_hot(y_train, num_classes)
    y_test  = one_hot(y_test, num_classes)

    return X_train, X_test, y_train, y_test