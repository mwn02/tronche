import numpy as np
import math
from network.from_scratch.data_processing import load_mnist_data

class Network():
    def __init__(self, layers, dtype=np.float32):
        self.layers = layers
        self.dtype = dtype
        self.weights = [np.random.randn(layers[i+1], layers[i]).astype(dtype) * np.sqrt(2 / layers[i]) for i in range(len(layers)-1)] # He initialization
        self.biases = [np.zeros(l, dtype=dtype) for l in layers[1:]]

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_prime(self, Z):
        return (Z > 0).astype(float) # quand x = 0, ça devrait être undefined, mais il faut faire un choix pratique
    
    def softmax(self, Z):
        exp_z = np.exp(Z)
        return exp_z / (np.sum(exp_z))
    
    def cost_prime(self, predictions, label):
        return predictions - label

    def is_prediction_good(self, output_layer, label):
        return label[np.argmax(output_layer)] == 1
    
    def forward(self, x, train=False):
        self.sums = []
        self.activations = []

        sums = [np.zeros(l, dtype=self.dtype) for l in self.layers]
        activations = [np.zeros(l, dtype=self.dtype) for l in self.layers]
        activations[0] = x.flatten()

        for l in range(len(self.layers)-1):
            sums[l] = self.weights[l] @ activations[l] + self.biases[l]
            if l == len(self.layers) - 2:
                activations[l+1] = sums[l] # pas appliquer d'activation aux valeurs de la couche finale
            else:
                activations[l+1] = self.relu(sums[l])

        predictions = self.softmax(activations[-1])

        if train:
            self.sums = sums
            self.activations = activations

        return predictions
    
    def train(self, X_train, X_test, y_train, y_test, batch_size, num_epochs, lr, test_logs=True):
        num_batch = math.ceil(len(X_train) / batch_size)

        for epoch in range(num_epochs):

            # mélanger les données avant la création de nouveaux mini-lots
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            for batch in range(num_batch):
                start_index = batch*batch_size
                end_index = min((batch+1)*batch_size, len(X_train))
                real_batch_size = float(end_index - start_index)

                xs = X_train[start_index:end_index]
                labels = y_train[start_index:end_index]

                gradient_w = [np.zeros((self.layers[i+1], self.layers[i]), dtype=self.dtype) for i in range(len(self.layers)-1)]
                gradient_b = [np.zeros(l, dtype=self.dtype) for l in self.layers[1:]]

                for x, label in zip(xs, labels):
                    predictions = self.forward(x, True)

                    # Propagation arrière
                    error = self.cost_prime(predictions, label)
                    for l in range(len(self.layers)-2, -1, -1):
                        gradient_w[l] += error[:,None] @ self.activations[l][None,:]
                        gradient_b[l] += error.copy()
                        if l != 0:
                            error = (self.weights[l].T @ error) * self.relu_prime(self.sums[l-1])

                # Descente du gradient
                for i in range(len(self.layers)-1):
                    self.weights[i] -= lr * gradient_w[i] / real_batch_size
                    self.biases[i] -= lr * gradient_b[i] / real_batch_size

            if test_logs:
                print(f"---------- Epoch #{epoch+1} ----------")
                self.test(X_test, y_test)

    def test(self, X_test, y_test):
        accuracy = 0
        for x, label in zip(X_test, y_test):
            prediction = self.forward(x)
            accuracy += 1 if self.is_prediction_good(prediction, label) else 0
        accuracy /= len(X_test)
        print(f"Model accuracy: {accuracy*100:.2f}%")
    def save_model(self, path):
        data = {}
        for i, W in enumerate(self.weights):
            data[f"W{i}"] = W
        for i, b in enumerate(self.biases):
            data[f"B{i}"] = b
        data["layers"] = self.layers
        np.savez(path, **data)

    def load_model(self, path):
        data = np.load(path)
        self.layers = data["layers"]

        self.weights = []
        self.biases = []

        for i in range(len(self.layers)-1):
            self.weights.append(data[f"W{i}"])
            self.biases.append(data[f"B{i}"])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_data()
    network = Network([28*28, 128, 32, 10])
    # network.load_model("network/from_scratch/mnist_model.npz")
    network.train(X_train, X_test, y_train, y_test, 128, 10, 0.01)
    network.save_model("network/from_scratch/mnist_model.npz")