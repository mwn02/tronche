import numpy as np
from PIL import Image, ImageOps
import random
import os

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

print(f"Dataset Prep Complete!")
print(f"Images shape: {X.shape}") # Should be (Number_of_Images, 32, 32, 1)
print(f"Labels shape: {y.shape}")


    

















# --- 1. ACTIVATION FUNCTIONS ---
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Softmax:
    def forward(self, x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- 2. LAYERS ---
class Dense:
    def __init__(self, input_size, output_size, lr=0.01):
        # He Initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.bias = np.zeros((1, output_size))
        self.lr = lr
        self.v_w = 0 # Momentum
        self.v_b = 0

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, momentum=0.9):
        grad_w = np.dot(self.input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        
        # Update with Momentum
        self.v_w = momentum * self.v_v_w - self.lr * grad_w
        self.v_b = momentum * self.v_b - self.lr * grad_b
        self.weights += self.v_w
        self.bias += self.v_b
        
        return np.dot(grad_output, self.weights.T)

# --- 3. THE FULL NETWORK ---
class ScratchCNN:
    def __init__(self):
        # Replicating your File 2 architecture
        # Note: Conv logic is simplified to Linear for brevity in this example 
        # but follows the same forward/backward pattern.
        self.layers = [
            Dense(32*32, 512), # Flattened 32x32 input
            ReLU(),
            Dense(512, 512),
            ReLU(),
            Dense(512, 5),    # 5 Output classes
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

# --- 4. TRAINING LOGIC (From File 1) ---
def cross_entropy_loss(pred, real):
    n_samples = real.shape[0]
    logp = -np.log(pred[range(n_samples), real])
    return np.sum(logp) / n_samples

def train(model, x_train, y_train, epochs=50):
    for epoch in range(epochs):
        # Forward
        probs = model.forward(x_train)
        loss = cross_entropy_loss(probs, y_train)
        
        # Initial Gradient (Derivative of Cross Entropy + Softmax)
        grad = probs
        grad[range(len(y_train)), y_train] -= 1
        grad /= len(y_train)
        
        # Backward
        model.backward(grad)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Usage
# model = ScratchCNN()
# train(model, x_data, y_labels)