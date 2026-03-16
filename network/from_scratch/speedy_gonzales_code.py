import json
import numpy as np
from network.from_scratch.data_processing import load_mnist_data
import time
from multiprocessing import Pool, Process

class ConvLayer:
    def __init__(self, channels_out, filter_size):
        self.channels_out = channels_out
        self.filter_size = filter_size
        self.filters = np.random.randn(channels_out, filter_size, filter_size) * np.sqrt(2/ (filter_size * filter_size))
        self.biases = np.zeros(channels_out)
        self.bias_gradients = np.zeros_like(self.biases)
        self.filter_weights_gradients = np.zeros_like(self.filters)

    def forward(self, input_train):
        self.input_shape = input_train.shape
        self.input = input_train
        # biases of shape (1, C_out, 1, 1) (-1 infers the dimensions C_out automatically)
        return self.convolution(input_train, self.filters) + self.biases.reshape(1, -1, 1, 1)

    def backward(self, incoming_error): 
        # filter: C_in x F x H_out x W_out
        # previous activations (self.input): B x C_in x H_in x W_in
        # error: B x C_out x H_out x W_out
        batch_size, C_out, H, W = incoming_error.shape
        self.bias_gradients += incoming_error.sum(axis=(0, 2, 3)) # this leaves the shape (F,)
        for b in range(batch_size):
            self.filter_weights_gradients += self.convolution((self.input[b])[np.newaxis,:], incoming_error[b]).sum(axis=0)

        flipped = np.rot90(self.filters, 2, axes=(-2, -1))
        pad = self.filter_size - 1
        padded_error = np.pad(
            incoming_error,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0
        )
        backward_conv = self.convolution(padded_error, flipped).sum(axis=1)
        backward_conv = np.repeat(backward_conv[:, np.newaxis, :, :], self.input_shape[1], axis=1)
        return backward_conv
    
    def convolution(self, input, filter):
        batch_size, C_in, H, W = input.shape
        C_out, K_H, K_W = filter.shape
        out_h = H - K_H + 1
        out_w = W - K_W + 1
        patches = []
        for i in range(out_h):
            for j in range(out_w):
                patch = input[:, :, i:i+K_H, j:j+K_W]
                patches.append(patch.reshape(batch_size, C_in, -1)) # size of (B, C_in, k*k)

        # patches is now of size: out_h*out_w X B X C_in X k*k
        X = np.stack(patches, axis=-1) # size: B X C_in X k*k X out_h*out_w (it moved out_h*out_w at the end)
        F = filter.reshape(C_out, -1) # size of (C_out, k*k)
        out = np.einsum('ck,bdko->bco', F, X) # f: num filters, c: num channels, b: num batches, p: num patches
        return out.reshape(batch_size, C_out, out_h, out_w)
    
    def update(self, batch_size, learning_rate):  
        self.filters -= (learning_rate / batch_size) * self.filter_weights_gradients
        self.biases -= (learning_rate / batch_size) * self.bias_gradients
        self.filter_weights_gradients.fill(0)
        self.bias_gradients.fill(0)
    
    def get_gradients(self):
        return [self.filter_weights_gradients, self.bias_gradients]

    def add_gradients(self, gradients):
        self.filter_weights_gradients += gradients[0]
        self.bias_gradients += gradients[1]

    def get_params(self):
        return (self.channels_out, self.filter_size, self.filters, self.biases)

    def set_params(self, params):
        self.channels_out, self.filter_size, self.filters, self.biases = params

class MaxPoolingLayer: 
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.max_indices = None
        self.stride = stride

    def forward(self, input, stride):
        self.stride = stride
        self.input_shape = input.shape
        batch_size, num_filters, height, width = input.shape

        out_h = (height-self.pool_size) // stride + 1
        out_w = (width-self.pool_size) // stride + 1
        output = np.zeros((batch_size, num_filters, out_h, out_w))
        self.max_indices = np.zeros((batch_size, num_filters, out_h, out_w), dtype=int)

        for i in range(0, out_h):
            for j in range(0, out_w):
                start_h, start_w = i * self.stride, j * self.stride
                region = input[:, :, start_h:start_h+self.pool_size, start_w:start_w+self.pool_size]
                region_flat = region.reshape(batch_size, num_filters, -1)

                self.max_indices[:, :, i, j] = np.argmax(region_flat, axis=-1)
                output[:, :, i, j] = np.max(region_flat, axis=-1)

        return output
    
    def backward(self, incoming_error):
        batch_size, num_filters, error_height, error_width = incoming_error.shape
        error_input = np.zeros(self.input_shape)

        # Create index grids for the batch and channel dimensions so NumPy can pair
        # each (b, c) location with its corresponding spatial index (input_i, input_j).
        # These arrays broadcast to shape (B, C), allowing elementwise indexing:
        # error_input[b, c, input_i[b,c], input_j[b,c]].
        b_idx = np.arange(batch_size)[:, None] # (B, 1)
        c_idx = np.arange(num_filters)[None, :] # (1, C)
        for i in range(error_height):
            for j in range(error_width):
                max_index = self.max_indices[:, :, i, j]
                input_i = i * self.stride + max_index // self.pool_size
                input_j = j * self.stride + max_index % self.pool_size
                error_input[b_idx, c_idx, input_i, input_j] += incoming_error[:, :, i, j]
        return error_input
    
    def update(self, batch_size, learning_rate):   
        return None

    def get_gradients(self):
        return None
    
    def add_gradients(self, gradients):
        return None
    
    def get_params(self):
        return (self.pool_size, self.stride)

    def set_params(self, params):
        self.pool_size, self.stride = params
    
class Relu: 
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)
    
    def backward(self, incoming_error):
        relu_derivative = (self.input > 0).astype(float)
        return incoming_error * relu_derivative

    def update(self, batch_size, learning_rate): 
        return None
    
    def get_gradients(self):
        return None
    
    def add_gradients(self, gradients):
        return None
    
    def get_params(self):
        return None

    def set_params(self, params):
        return None
    
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1) # keep the batch size, flatten the rest
    
    def backward(self, incoming_error):
        return incoming_error.reshape(self.input_shape)
    
    def update(self, batch_size, learning_rate): 
        return None

    def get_gradients(self):
        return None
    
    def add_gradients(self, gradients):
        return None

    def get_params(self):
        return None

    def set_params(self, params):
        return None
    
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2/ input_size)
        self.biases = np.zeros(output_size)
        self.error = None
        self.dw_acc = np.zeros_like(self.weights) #pour faire l'accumulation de l'erreur sur les poids
        self.db_acc = np.zeros_like(self.biases)  # pour faire l'accumulation de l'erreur sur les biais

    def forward(self, input):
        self.input = input
        return self.input @ self.weights.T + self.biases
    
    def backward(self, incoming_error):
        # dZ : (B, out_dim)
        
        # Calculate gradients for this specific image
        weight_gradient = incoming_error.T @ self.input
        bias_gradient = incoming_error.sum(axis=0) 
        
        # ACCUMULATE: This is the secret sauce for batching
        self.dw_acc += weight_gradient
        self.db_acc += bias_gradient
        
        # Pass the error to the previous layer
        previous_layer_error = incoming_error @ self.weights
        return previous_layer_error
    
    def update(self, batch_size, learning_rate): 
        self.weights -= (learning_rate/batch_size) *self.dw_acc
        self.biases -= (learning_rate/batch_size) * self.db_acc

        self.dw_acc.fill(0)
        self.db_acc.fill(0)
        return self.weights, self.biases
    
    def get_gradients(self):
        return [self.dw_acc, self.db_acc]
    
    def add_gradients(self, gradients):
        self.dw_acc += gradients[0]
        self.db_acc += gradients[1]
    
    def get_params(self):
        return (self.input_size, self.output_size, self.weights, self.biases)

    def set_params(self, params):
        self.input_size, self.output_size, self.weights, self.biases = params
    
class CrossEntropyLoss:
    def __init__(self): 
        self.predictions = None
        self.real_value = None

    def forward(self, predictions, real_value): 
        self.predictions = predictions
        self.real_value = real_value
        eps = 67e-12
        return -np.sum(real_value*np.log(predictions+eps))
    
    def backward(self, predictions, real_value):
        return predictions - real_value
    
    def update(self, batch_size, learning_rate): 
        return None

    def get_gradients(self):
        return None
    
    def add_gradients(self, gradients):
        return None
    
    def get_params(self):
        return None

    def set_params(self, params):
        return None

def test_model(input_test, labels_test, layers):
    x = input_test.reshape(input_test.shape[0], 1, 28, 28) # batch size of 1 # x est l'image | est-ce que ce chang. de dims est nécessaire?
    labels = labels_test.reshape(labels_test.shape[0], 10) # batch size of 1 #change for emojis!

    # Handle if label is one-hot or an integer
    if labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    for layer in layers:
        if isinstance(layer, MaxPoolingLayer):
            x = layer.forward(x, stride=2) # mettre un pas de 2
        else:
            x = layer.forward(x)
            
    return np.sum([np.argmax(softmax(x), axis=1) == labels]) / input_test.shape[0]

def save_model(layers, accuracy, filename="model"):
    model_data = [] # Changed to list
    
    for layer in layers:
        # Use isinstance to check the class type
        if isinstance(layer, ConvLayer):
            model_data.append({
                'type': 'ConvLayer',
                'num_filters': layer.channels_out,
                'filter_size': layer.filter_size,
                'filters': layer.filters.tolist(),
                'biases': layer.biases.tolist()
            })
        elif isinstance(layer, DenseLayer):
            model_data.append({
                'type': 'DenseLayer',
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist()
            })
        elif isinstance(layer, MaxPoolingLayer):
            model_data.append({
                'type': 'MaxPoolingLayer',
                'pool_size': layer.pool_size,
                'stride': layer.stride
            })
        elif isinstance(layer, Relu):
            model_data.append({'type': 'Relu'})
        elif isinstance(layer, Flatten):
            model_data.append({'type': 'Flatten'})

    # Construct filename and save as JSON
    full_filename = f"{filename}_acc_{accuracy:.2f}.json"
    with open(full_filename, 'w') as f:
        json.dump(model_data, f)
    
    print(f"Model saved as {full_filename}")

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def one_hot(labels, num_classes):
# Ensure labels is a NumPy array of integers
    labels = np.array(labels).astype(int)
    return np.eye(num_classes)[labels]

def main():
    # Pour exécuter le script, taper dans le terminal: py -m network.from_scratch.tomas_speed
    #X, y = get_emoji_data("dataset/dataset-data/training-data/")
    #input_train, input_test, label_train, label_test = get_shuffled_data(X, y, 5, 0.8)

    # données
    input_train, input_test, label_train, label_test = load_mnist_data()
    # num_train = 30000
    # num_test = 5000
    # input_train = input_train[:num_train]
    # input_test = input_test[:num_test]
    # label_train = label_train[:num_train]
    # label_test = label_test[:num_test]

    # paramètres
    learning_rate = 0.01
    batch_size = 256
    num_epochs = 5

    # réseau
    # add reshape layer?
    layers = [
        ConvLayer(8, 3), # output shape: 26x26x8
        Relu(),         # output shape: 26x26x8
        MaxPoolingLayer(2, stride=2), # output shape: 13x13x8
        ConvLayer(16, 3),   # output shape: 11x11x16
        Relu(),            # output shape: 11x11x16
        MaxPoolingLayer(2, stride=2), # output shape: 5x5x16
        Flatten(),         # output shape: 400
        DenseLayer(400, 100),
        Relu(),
        DenseLayer(100, 10),
    ]
    # loss function chosen in the worker()

    timestamp = time.time()
    n_workers = 15
    with Pool(n_workers) as p:
        # entrainement par époques
        for epoch in range(num_epochs):
            # mélanger les données
            perm = np.random.permutation(len(input_train))
            input_train_shuffled = input_train[perm]
            label_train_shuffled = label_train[perm]

            # entrainement par mini-lots
            for start in range(0, len(input_train), batch_size):
                layer_params = get_layer_params(layers)
                end = min(start + batch_size, len(input_train))      
                real_batch_size = end - start
                # entrainement d'un mini-lot
                inputs = input_train_shuffled[start:end]
                labels = label_train_shuffled[start:end]
                x = inputs.reshape(real_batch_size, 1, 28, 28) # x est l'image | est-ce que ce chang. de dims est nécessaire?
                labels = labels.reshape(real_batch_size, 10) # change for emojis!

                chunk_size = real_batch_size // n_workers + 1
                params = []
                for i in range(0, real_batch_size, chunk_size):
                    j = min(i + chunk_size, real_batch_size)  
                    new_x = x[i:j]
                    new_labels = labels[i:j]
                    params.append((layer_params, new_x, new_labels))

                results = p.starmap(worker, params)
            
                for result in results:
                    add_gradients(layers, result)
                # fin de l'entrainement d'un mini-lot: descente du gradient
                for layer in layers: 
                    layer.update(real_batch_size, learning_rate)

                print(f"\rEpoch {epoch+1} | Batch {start//batch_size + 1} processing...", end="")
            
            # tester le modèle
            accuracy = test_model(input_test, label_test, layers)
            print(f"\nEpoch {epoch+1} Done! Test Accuracy: {accuracy*100:.2f}%")

    print(f"finished in {time.time() - timestamp}")

def worker(layer_params, x, labels):
    layers = [
        ConvLayer(8, 3), # output shape: 26x26x8
        Relu(),         # output shape: 26x26x8
        MaxPoolingLayer(2, stride=2), # output shape: 13x13x8
        ConvLayer(16, 3),   # output shape: 11x11x16
        Relu(),            # output shape: 11x11x16
        MaxPoolingLayer(2, stride=2), # output shape: 5x5x16
        Flatten(),         # output shape: 400
        DenseLayer(400, 100),
        Relu(),
        DenseLayer(100, 10),
    ]
    set_layer_params(layers, layer_params)
    for layer in layers:
        if isinstance(layer, MaxPoolingLayer):
            x = layer.forward(x, stride=2) # mettre un pas de 2
        else:
            x = layer.forward(x)
            
    predictions = softmax(x)

    loss = CrossEntropyLoss()
    loss.forward(predictions, labels)
    error = loss.backward(predictions, labels)

    # rétropropagation
    for layer in reversed(layers):
        error = layer.backward(error)
    
    return get_gradients(layers)

def get_gradients(layers):
    gradients = []
    for layer in layers:
        gradients.append(layer.get_gradients())
    return gradients

def add_gradients(layers, gradients):
    for layer, gradient in zip(layers, gradients):
        layer.add_gradients(gradient)
    
def get_layer_params(layers):
    params = []
    for layer in layers:
        params.append(layer.get_params())
    return params

def set_layer_params(layers, params):
    for layer, param in zip(layers, params):
        layer.set_params(param)

# train a single batch
# params: layers, inputs, labels, lr,

if __name__ == "__main__":
    main()
    