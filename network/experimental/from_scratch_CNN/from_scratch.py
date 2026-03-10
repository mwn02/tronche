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

# 4. Convert lists to NumPy arrays and shuffle
X = np.array(all_images)
y = np.array(all_labels)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

train_data = X[:split]
train_labels = y[:split]

test_data = X[split:]
test_labels = y[split:]

num_classes = len(np.unique(y))

y_train = np.eye(num_classes)[train_labels]   # One-hot encoding pour bien marcher avec softmax et cross-entropy
y_test = np.eye(num_classes)[test_labels]

#on commence à construire le CNN 

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9) #on ajoute un petit nombre pour éviter log(0)

#convolution layer
# --- FIX: Define variables before using them ---
num_filters = 16
filter_size = 3

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2/ (filter_size * filter_size))

    def forward_convolution(self, input):
        self.input = input
        # --- FIX: Must unpack 3 values (h, w, channels) ---
        height, width, _ = input.shape 
        
        output = np.zeros((height - self.filter_size + 1,
                           width - self.filter_size + 1,
                           self.num_filters))

        for f in range(self.num_filters):
            for i in range(height - self.filter_size + 1):
                for j in range(width - self.filter_size + 1):
                    region = input[i:i+self.filter_size, j:j+self.filter_size, 0]
                    output[i, j, f] = np.sum(region * self.filters[f])
        return output

    def backward_convolution(self, d_output, learning_rate):
        derive_filters = np.zeros_like(self.filters)
        height, width, _ = self.input.shape

        for f in range(self.num_filters):
            for i in range(height - self.filter_size + 1):
                for j in range(width - self.filter_size + 1):
                    region = self.input[i:i+self.filter_size, j:j+self.filter_size, 0]
                    derive_filters[f] += d_output[i, j, f] * region
        
        self.filters -= learning_rate * derive_filters
        return self.filters

    def backward_convolution(self, d_output, learning_rate):
        input = ... ###########################################
        self.input = input
        derive_filters = np.zeros_like(self.filters)
        height, width, _ = self.input.shape

        for f in range(self.num_filters):
            for i in range(height - self.filter_size + 1):
                for j in range(width - self.filter_size + 1):
                    region = self.input[i:i+self.filter_size, j:j+self.filter_size,0]
                    derive_filters[f] += d_output[i, j, f] * region

        # Update filters
        self.filters -= learning_rate * derive_filters
        return self.filters
    
    def pooling_layer(self, x, size=2, stride=2):
        x = self.forward_convolution(num_filters=16, filter_size=3) #voir c<est quoi le resultat du conv layer pour faire le pooling
        self.pool_input = x
        h, w, num_filters = x.shape
        output_h = (h - size) // stride + 1    
        output_w = (w - size) // stride + 1
        output = np.zeros((output_h, output_w, num_filters))

        self.max_indices = [] # garder les indices des max pour la backprop

        for f in range(num_filters):
            for i in range(0, h - size + 1, stride):
                for j in range(0, w - size + 1, stride):

                    region = x[i:i+size, j:j+size, f]

                    max_index = np.unravel_index(np.argmax(region), region.shape) #unravel remet sous forme de "matrice 2d" les indices du max

                    output[i//stride, j//stride, f] = region[max_index]

                    # Save absolute position of max (for backprop)
                    self.max_indices[(i//stride, j//stride, f)] = (
                        i + max_index[0],
                        j + max_index[1]
                    )

        return output
