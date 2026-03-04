# Pour exécuter le script, taper dans le terminal: py -m network.main.from_scratch.main

import numpy as np
from network.main.from_scratch.data_processing import get_network_data
import math

################## DONNÉES ##################
base_path = "dataset/dataset-data/training-data/"
train_data, test_data, train_labels, test_labels = get_network_data(base_path, 0.8)

################## PARAMÈTRES ##################
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Couche d'entrée (image aplatie), couches cachées, couche de sortie (5 classes)
layers = [32*32, 128, 64, 5] 

# attribuer des poids aléatoires pour chaque couche
weights = [np.random.randn(layers[i+1], layers[i]) * np.sqrt(2 / layers[i]) for i in range(len(layers)-1)] # He initialization
# PARLER DE HE INITIALIZATION SI ON L'UTILISE, SINON, JUSTE NP.RANDOM entre [-1, 1]
# attribuer des biais aléatoires pour chaque couche
bias = [np.zeros(l) for l in layers[1:]]

################## RÉSEAU ##################
def relu(Z):
    return np.maximum(0, Z)
def relu_prime(Z):
    return (Z > 0).astype(float) # quand x = 0, ça devrait être undefined, mais il faut faire un choix pratique
def softmax(Z):
    exp_logits = np.exp(Z)
    return exp_logits / np.sum(exp_logits)
def cost_prime(predictions, label):
    return predictions - label

def test():
    def forward(x):
        # Propagation avant
        sums = [np.zeros(l, dtype=np.float32) for l in layers]
        activations = [np.zeros(l, dtype=np.float32) for l in layers]
        activations[0] = x.flatten()

        for l in range(len(layers)-1):
            sums[l] = weights[l] @ activations[l] + bias[l]
            if l == len(layers) - 2:
                activations[l+1] = sums[l] # pas appliquer d'activation aux valeurs de la couche finale
            else:
                activations[l+1] = relu(sums[l])

        predictions = softmax(activations[-1])
        return predictions


    def is_prediction_good(output_layer, label):
        return label[np.argmax(output_layer)] == 1

    accuracy = 0
    for x, label in zip(test_data, test_labels):
        prediction = forward(x)
        accuracy += 1 if is_prediction_good(prediction, label) else 0
    accuracy /= len(test_data)
    print(f"Model accuracy: {accuracy*100:.2f}%")


num_batch = math.ceil(len(train_data) / batch_size)

for epoch in range(num_epochs):
    print(f"---------- Epoch #{epoch+1} ----------")
    for batch in range(num_batch):
        start_index = batch*batch_size
        end_index = min((batch+1)*batch_size, len(train_data))
        real_batch_size = float(end_index - start_index)

        xs = train_data[start_index:end_index]
        labels = train_labels[start_index:end_index]

        gradient_w = [np.zeros((layers[i+1], layers[i])) for i in range(len(layers)-1)]
        gradient_b = [np.zeros(l) for l in layers[1:]]

        for x, label in zip(xs, labels):
            # Propagation avant
            sums = [np.zeros(l, dtype=np.float32) for l in layers]
            activations = [np.zeros(l, dtype=np.float32) for l in layers]
            activations[0] = x.flatten()

            for l in range(len(layers)-1):
                sums[l] = weights[l] @ activations[l] + bias[l]
                if l == len(layers) - 2:
                    activations[l+1] = sums[l] # pas appliquer d'activation aux valeurs de la couche finale
                else:
                    activations[l+1] = relu(sums[l])

            predictions = softmax(activations[-1])

            # Propagation arrière
            error = cost_prime(predictions, label)
            for l in range(len(layers)-2, -1, -1):
                gradient_w[l] += error[:,None] @ activations[l][None,:]
                gradient_b[l] += error.copy()
                if l != 0:
                    error = (weights[l].T @ error) * relu_prime(sums[l-1])

        # Descente du gradient
        for i in range(len(layers)-1):
            weights[i] -= learning_rate * gradient_w[i] / real_batch_size
            bias[i] -= learning_rate * gradient_b[i] / real_batch_size
    test()

# save model avec np.save