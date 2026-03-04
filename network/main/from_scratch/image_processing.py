import numpy as np
from PIL import Image, ImageOps
import random
import os

def get_network_data(base_path, data_ratio):
    """
    Retourne les données d'entrainement et de test, pour un dossier d'images donné.

    La valeur retournée est: training_data, test_data, training_labels, test_labels.
    Les étiquettes sont des vecteurs indicateurs (one-hot vectors)
    
    Le data_ratio est le % des données allouées au training_data, et le
    reste au test_data
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

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split = int(data_ratio * len(X))

    train_data = X[:split]
    train_labels = y[:split]

    test_data = X[split:]
    test_labels = y[split:]

    num_classes = len(np.unique(y))

    # One-hot encoding pour bien marcher avec softmax et cross-entropy
    train_labels_hot = np.eye(num_classes)[train_labels]
    test_labels_hot = np.eye(num_classes)[test_labels]

    return train_data, test_data, train_labels_hot, test_labels_hot