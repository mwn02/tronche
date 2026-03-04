# Pour exécuter le script, taper dans le terminal: py -m network.main.from_scratch.main

import numpy as np
from network.main.from_scratch.image_processing import get_network_data

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

################## NETWORK ##################
def relu(Z):
    return np.maximum(0, Z)
def relu_prime(Z):
    return (Z > 0).astype(float) # quand x = 0, ça devrait être undefined, mais il faut faire un choix pratique
def softmax(Z):
    exp_logits = np.exp(Z)
    return exp_logits / np.sum(exp_logits)
def cost_prime(A, Y):
    # empêcher les erreurs de division par 0
    eps = 1e-12
    A = np.clip(A, eps, 1 - eps)
    return (A - Y) / (A * (1 - A))
  
# Propagation avant
sums = [np.zeros(l, dtype=np.float32) for l in layers]
activations = [np.zeros(l, dtype=np.float32) for l in layers]
activations[0] = train_data[0].flatten()
label = train_labels[0]

for l in range(len(layers)-1):
    sums[l] = weights[l] @ activations[l] + bias[l]
    activations[l+1] = relu(sums[l])

predictions = softmax(activations[-1])
print("Predicted probabilities:", predictions)

# Propagation arrière
gradient_w = [np.zeros((layers[i+1], layers[i])) for i in range(len(layers)-1)]
gradient_b = [np.zeros(l) for l in layers[1:]]
# preciser le dtype??

error = cost_prime(activations[-1], label)
for l in range(len(layers)-2, -1, -1):
    gradient_w[l] += error[:,None] @ activations[l][None,:]
    gradient_b[l] += error.copy()
    if l != 0:
        error = (weights[l].T @ error) * relu_prime(sums[l-1])

# Descente du gradient
for i in range(len(layers)-1):
    weights[i] -= learning_rate * gradient_w[i] / batch_size
    bias[i] -= learning_rate * gradient_b[i] / batch_size

# CE QU'IL RESTE À FAIRE:
# Mettre tout le processus dans une loop d'epoch et de batch
# il faut donc aussi créer les batchs avant de loops chaque batch