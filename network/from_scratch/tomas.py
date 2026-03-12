
from email.mime import image

import numpy as np
import math
from network.experimental.from_scratch_CNN.from_scratch import relu_derivative
from network.from_scratch.data_processing import load_mnist_data

# Pour exécuter le script, taper dans le terminal: py -m network.from_scratch.tomas
#X, y = get_emoji_data("dataset/dataset-data/training-data/")
#input_train, input_test, label_train, label_test = get_shuffled_data(X, y, 5, 0.8)
input_train, input_test, label_train, label_test = load_mnist_data()
input_train = input_train[:300]
label_train = label_train[:300]
input_test = input_test[:100]
label_test = label_test[:100]
learning_rate = 0.01
batch_size = 32
num_epochs = 10
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2/ (filter_size * filter_size))
        self.biases = np.zeros(num_filters)
        self.bias_gradients = np.zeros_like(self.biases)
        self.filter_weights_gradients = np.zeros_like(self.filters)


    def forward(self, input_train):
        self.input = input_train
        height, width, _ = input_train.shape 

        output = np.zeros((height - self.filter_size + 1,
                        width - self.filter_size + 1,
                        self.num_filters))
        
        for f in range(self.num_filters):
            for i in range(height - self.filter_size + 1):
                for j in range(width - self.filter_size + 1):
                    region = input_train[i:i+self.filter_size, j:j+self.filter_size, 0]
                    output[i, j, f] = np.sum(region * self.filters[f]) + self.biases[f]
        return output
    

    def backward(self, incoming_error): 
        incoming_error_height, incoming_error_width, num_filters = incoming_error.shape[0], incoming_error.shape[1], incoming_error.shape[2]
        previous_layer_error = np.zeros_like(self.input)
        for f in range(num_filters):
            self.bias_gradients[f] += np.sum(incoming_error[:, :, f])
            for i in range(self.filter_size):
                for j in range(self.filter_size):
                    region = self.input[i:i+incoming_error_height, j:j+incoming_error_width, 0]
                    self.filter_weights_gradients[f, i, j] += np.sum(region * incoming_error[:, :, f])
            for i in range(incoming_error_height):
                for j in range(incoming_error_width):
                    previous_layer_error[i:i+self.filter_size, j:j+self.filter_size, 0] += self.filters[f] * incoming_error[i, j, f]
        return previous_layer_error         
    
    def update(self, batch_size, learning_rate):  
        self.filters -= (learning_rate / batch_size) * self.filter_weights_gradients
        self.biases -= (learning_rate / batch_size) * self.bias_gradients
        

        self.filter_weights_gradients.fill(0)
        self.bias_gradients.fill(0)
        return None
    
class MaxPoolingLayer: 
    def __init__(self,pool_size,stride=2):
        self.pool_size = pool_size
        self.input = None
        self.max_indices = None
        self.stride = 2

    def forward(self, input, stride):
        self.stride = stride
        self.input = input
        height, width, num_filters = input.shape
        output = np.zeros(((height-self.pool_size) // stride + 1, (width-self.pool_size) // stride + 1, num_filters))
        self.max_indices = np.zeros_like(output, dtype=int)
        for filters in range(num_filters): 
            for i in range(0, height - self.pool_size + 1, stride):
                for j in range(0, width - self.pool_size + 1, stride):
                    region = input[i:i+self.pool_size, j:j+self.pool_size, filters]
                    max_index = region.argmax()
                    output_i, output_j = i // stride, j // stride
                    output[output_i, output_j, filters] = np.max(region)
                    self.max_indices[output_i, output_j, filters] = max_index    # [0,1], [2,3] sont les valeurs que prendraient les indices max
        return output
    def backward(self,incoming_error):  
        num_filters = self.input.shape[2]
        error_input = np.zeros_like(self.input)

        error_input = np.zeros_like(self.input)
        for filters in range(num_filters): 
            for i in range(incoming_error.shape[0]):
                for j in range(incoming_error.shape[1]):
                    max_index = self.max_indices[i, j, filters]
                    input_i = i * self.stride + max_index // self.pool_size
                    input_j = j * self.stride + max_index % self.pool_size
                    error_input[input_i, input_j, filters] += incoming_error[i, j, filters]
        return error_input
    
    def update(self, batch_size, learning_rate):   
        return None
    
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

class Flatten:
    def __init__(self):
        self.input_shape = None
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()
    def backward(self, incoming_error ): 
        return incoming_error.reshape(self.input_shape)
    def update(self, batch_size, learning_rate): 
        return None
    
class DenseLayer:
    def __init__(self,input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2/ input_size)
        self.biases = np.zeros(output_size)
        self.error = None
        self.dw_acc = np.zeros_like(self.weights) #pour faire l'accumulation de l'erreur sur les poids
        self.db_acc = np.zeros_like(self.biases)  # pour faire l'accumulation de l'erreur sur les biais

    def forward(self, input):
        self.input = input.reshape(-1,1)
        return (self.weights @ self.input + self.biases.reshape(-1, 1)).flatten()
    
    def backward(self, incoming_error):
        error_column_vector = incoming_error.reshape(-1, 1)
        
        # Calculate gradients for this specific image
        weight_gradient = error_column_vector @ self.input.T
        bias_gradient = incoming_error.flatten()  
        
        # ACCUMULATE: This is the secret sauce for batching
        self.dw_acc += weight_gradient
        self.db_acc += bias_gradient
        
        # Pass the error to the previous layer
        previous_layer_error = (self.weights.T @ error_column_vector).flatten()
        return previous_layer_error
    
    def update(self, batch_size, learning_rate): 
        self.weights -= (learning_rate/batch_size) *self.dw_acc
        self.biases -= (learning_rate/batch_size) * self.db_acc

        self.dw_acc.fill(0)
        self.db_acc.fill(0)
        return self.weights, self.biases

    
class CrossEntropyLoss:
    def __init__(self): 
        self.predictions = None
        self.real_value = None
    def forward(self, predictions,real_value): 
        self.predictions = predictions
        self.real_value = real_value
        eps = 67e-12
        return -np.sum(real_value*np.log(predictions+eps))
    def backward(self, predictions,real_value):
        return predictions - real_value
    def update(self, batch_size, learning_rate): 
        return None
    


test_image = input_train[0]
test_image = test_image.reshape(28, 28, 1) # reshape pour ajouter une dimension de canal
layers = [
    ConvLayer(8,3), # output shape: 26x26x8
    Relu(),         # output shape: 26x26x8
    MaxPoolingLayer(2,stride=2), # output shape: 13x13x8
    ConvLayer(16,3),   # output shape: 11x11x16
    Relu(),            # output shape: 11x11x16
    MaxPoolingLayer(2,stride=2), # output shape: 5x5x16
    Flatten(),         # output shape: 400
    DenseLayer(400,100),
    Relu(),
    DenseLayer(100,10),
]
loss = CrossEntropyLoss()
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

for epoch in range(num_epochs):

    perm = np.random.permutation(len(input_train))
    input_train_shuffled = input_train[perm]
    label_train_shuffled = label_train[perm]

    for i in range(0, len(input_train), batch_size):
        start = i
        end = min(i + batch_size, len(input_train))      
        real_batch_size = end - start
        batch_input = input_train_shuffled[start:end]
        batch_labels = label_train_shuffled[start:end]
        for j in range(real_batch_size):
            image = batch_input[j].reshape(28,28,1)

            def one_hot(labels, num_classes):
            # Ensure labels is a NumPy array of integers
                labels = np.array(labels).astype(int)
                return np.eye(num_classes)[labels]
            
            label = one_hot(batch_labels[j],10)
            x=image

            for layer in layers:
                if isinstance(layer, MaxPoolingLayer):
                    x = layer.forward(x, stride=2) # Added stride=2
                    print(f"Layer {type(layer).__name__} output shape: {x.shape}")
                else:
                    x = layer.forward(x)
                    print(f"Layer {type(layer).__name__} output shape: {x.shape}")

            predictions = softmax(x)
            loss_value = loss.forward(predictions,label)
            error = loss.backward(predictions,label)

            for layer in reversed(layers):
                error = layer.backward(error)
                print(f"Layer {type(layer).__name__} output shape after backward: {error.shape}")

        for layer in layers: 
            layer.update(batch_size, learning_rate)

    print(f"Epoch {epoch} done")

    def test_model(input_test, label_test):
        correct_predictions = 0
        x=image
        for i in range(len(input_test)):
            image = input_test[i].reshape(28,28,1)
            label = label_test[i]

            x = image
            for layer in layers:
                x = layer.forward(x)

            predicted_label = np.argmax(x)
            if predicted_label == label:
                correct_predictions += 1

        accuracy = correct_predictions / len(input_test)
        print(f"Test Accuracy: {accuracy*100:.2f}%")




 
