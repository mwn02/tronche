import numpy as np 
from parameters import weights, biais
from image_processing import train_data, train_labels, test_data, test_labels



def relu(x):
    return np.maximum(0, x)
def softmax(val_final_layer):
    exp_logits = np.exp(val_final_layer - np.max(val_final_layer)) # Numerical stability
    return exp_logits / np.sum(exp_logits)


# ce code fonctionne pour les NN traditionnels, semblable au dense layer
val_input = train_data.flatten() # logiquement on devrait avoir des differentes dimensions si on faisait un CNN du au max pooling.
val_couche1 = relu(val_input@weights[0] + biais[0]) 
val_couche2 = relu(val_couche1@weights[1] + biais[1])
val_final_layer = val_couche2@weights[2] + biais[2]
predictions = softmax(val_final_layer)
print("Predicted probabilities:", predictions)
