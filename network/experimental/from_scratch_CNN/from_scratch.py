import numpy as np
from PIL import Image, ImageOps
import random
import os
#to play code :   py -m network.experimental.from_scratch_CNN.from_scratch

# Base path from your screenshot
base_path = "dataset/dataset-data/training-data/"

# --- NEW: Initialize storage for your CNN data ---
all_images = []
all_labels = []

# 1. Loop through category folders (0, 1, 2, 3, 4)
for label_folder in os.listdir(base_path):
    category_path = os.path.join(base_path, label_folder)
    
    if not os.path.isdir(category_path):
        continue

    # 2. Loop through actual images in those folders
    for image_name in os.listdir(category_path):
        full_path = os.path.join(category_path, image_name)
        
        try:
            img = Image.open(full_path)
            img = img.resize((32, 32), resample=Image.BILINEAR)
            img = ImageOps.grayscale(img)

            # Data Augmentation (Rotation + Translation)
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
            
            # --- NEW: Append to our dataset ---
            # We add a channel dimension so shape becomes (32, 32, 1)
            img_np = np.expand_dims(img_np, axis=-1) 
            
            all_images.append(img_np)
            all_labels.append(int(label_folder)) # Assumes folder names are 0, 1, 2...

        except Exception as e:
            print(f"Skipping {image_name}: {e}")

# --- NEW: Final Conversion for CNN ---
X = np.array(all_images)
y = np.array(all_labels)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

num_classes = len(np.unique(y))

y_train = np.eye(num_classes)[y_train]
y_val = np.eye(num_classes)[y_val]

#start building CNN from scratch here using X_train, y_train, X_val, y_val

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)

#convolution layer
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2/ (filter_size * filter_size))             
    def forward(self, input):
        self.input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1,
                           w - self.filter_size + 1,
                           self.num_filters))

        for f in range(self.num_filters):
            for i in range(h - self.filter_size + 1):
                for j in range(w - self.filter_size + 1):
                    region = input[i:i+self.filter_size, j:j+self.filter_size]
                    output[i, j, f] = np.sum(region * self.filters[f])

        return output
    
    def backward_vectorized(self, d_L_d_out, learning_rate):
    h_out, w_out, num_filters = d_L_d_out.shape
    f_size = self.filter_size
    
    d_L_d_filters = np.zeros(self.filters.shape)

    
    for i in range(f_size):
        for j in range(f_size):
            
            input_slice = self.input[i:i+h_out, j:j+w_out]
            for f in range(num_filters):
                d_L_d_filters[f, i, j] = np.sum(input_slice * d_L_d_out[:, :, f])

    self.filters -= learning_rate * d_L_d_filters
    return None  # No need to return anything for the convolution layer's backward pass
    

    














