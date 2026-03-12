
import numpy as np
import math
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
num_epochs = 20 
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2/ (filter_size * filter_size))
        self.biases = np.zeros(num_filters)
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
    def backwards(self, ): 
        return None
    
class MaxPoolingLayer: 
    def __init__(self,pool_size):
        self.pool_size = pool_size
        self.input = None
        self.max_indices = None

    def forward(self, input, stride):
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
                    self.max_indices[output_i, output_j, filters] = max_index               # [0,1], [2,3] sont les valeurs que prendraient les indices max
        return output
    def backwards(self, ): 
        return None
    
class Relu: 
    def __init__(self):
        self.input = None
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)
    def backwards(self, ): 
        return None

class Flatten:
    def __init__(self):
        self.input_shape = None
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()
    def backwards(self, incoming_error ): 
        return incoming_error.reshape(self.input_shape)
    
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
        self.input = input
        return self.weights @ self.input + self.biases
    
    def backward(self,incoming_error):
        self.incoming_error = incoming_error
        error_collumn_vector = incoming_error.reshape(-1,1)
        input_row_vector = self.input.reshape(1,-1)
        weight_gradient = error_collumn_vector@input_row_vector
        bias_gradient = incoming_error
        previous_layer_error = self.weights.T @ incoming_error
        return weight_gradient,bias_gradient,previous_layer_error
    
    def batch_backwards(self,weight_gradient,bias_gradient): 
        self.dw_acc += weight_gradient
        self.db_acc += bias_gradient
        return self.dw_acc, self.db_acc
    def update_weights_and_biases(self, batch_size, learning_rate): 
        self.weights -= (learning_rate/batch_size) *self.dw_acc
        self.biases -= (learning_rate/batch_size) * self.db_acc

        self.dw_acc.fill(0)
        self.db_acc.fill(0)
        return self.weights, self.biases

class Softmax: 
    def __init__(self):
        self.input = None
    def forward(self,input): 
        self.input = input
        exp_z = np.exp(input)
        return exp_z / (np.sum(exp_z))
    
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
        return self.predictions - self.real_value
    


test_image = input_train[0]
test_image = test_image.reshape(28, 28, 1) # reshape pour ajouter une dimension de canal
conv_layer1 = ConvLayer(8,3)
relu1 = Relu()
pool_layer1 = MaxPoolingLayer(2)
conv_layer2 = ConvLayer(16,3)
relu2 = Relu()
pool_layer2 = MaxPoolingLayer(2)
flattened = Flatten()
denselayer1 = DenseLayer(400,100)
relu_hidden1 = Relu()
denselayer2 = DenseLayer(100,10)
softmax = Softmax()
lossfn = CrossEntropyLoss()



for epoch in range(num_epochs): 
    perm = np.random.permutation(len(input_train))
    input_train_shuffled = input_train[perm]
    label_train_shuffled = label_train[perm]
    for i in range(0, len(input_train), batch_size):
        real_values = one_hot(label_train[i], 10) # one-hot encoding de la vraie étiquette



        conv_output1 = conv_layer1.forward(test_image)
        relu_output1 = relu1.forward(conv_output1)
        pool_output1 = pool_layer1.forward(relu_output1, stride=2)
        conv_output2 = conv_layer2.forward(pool_output1)
        relu_output2 = relu2.forward(conv_output2)
        pool_output2 = pool_layer2.forward(relu_output2, stride=2)
        flattened_output = flattened.forward(pool_output2)
        denselayer_output1 = denselayer1.forward(flattened_output)
        relu_hidden_output1 = relu_hidden1.forward(denselayer_output1)
        denselayer_output2 = denselayer2.forward(relu_hidden_output1)
        predictions = softmax.forward(denselayer_output2)
        error_output = lossfn.backward(predictions, real_values)  #find out how to get the real value
        denselayer_output2_gradient, denselayer_output2_bias_gradient, relu_hidden_output1_error = denselayer2.backward(error_output)
        denselayer2.batch_backwards(denselayer_output2_gradient, denselayer_output2_bias_gradient)
        denselayer2.update_weights_and_biases(batch_size, learning_rate)


incoming_error = 
 
