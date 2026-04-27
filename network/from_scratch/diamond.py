

from multiprocessing import Pool
from multiprocessing.dummy import Pool
import time

import time

from network.from_scratch.speedy_gonzales_code import ConvLayer, get_layer_params, Relu, MaxPoolingLayer, Flatten, DenseLayer, get_shuffled_data, worker, add_gradients, test_model, save_model
from network.with_pytorch_for_esp32.data_fetching import get_emoji_data
import random
import numpy as np  

def main():
    """
    Cette fonction est la partie centrale du code, elle gère l'entrainement du modèle. '
    Elle charge les données, définit les paramètres d'entrainement, crée le réseau de neurones, 
    et lance l'entrainement par époques. Après chaque époque, elle teste le modèle sur les données de test '
    et affiche l'accuracy. Enfin, elle sauvegarde le modèle entrainé.
    """

    # Pour exécuter le script, taper dans le terminal: py -m network.from_scratch.tomas_speed
    X, y = get_emoji_data("dataset/dataset-data/training-data/")
    input_train, input_test, label_train, label_test = get_shuffled_data(X, y, 5, 0.8)

    # paramètres
    learning_rate = 0.01
    mega_batch_size = 465
    num_epochs = 30

    layers = [            #initial est 32x32x1                                                
        ConvLayer(32, 3), # output shape: 32x30x30                          
        Relu(),         # output shape: 32x30x30                           
        MaxPoolingLayer(2, stride=2), # output shape: 32x15x15
        Flatten(),         # output shape: 7200
        DenseLayer(7200, 512),
        Relu(),
        DenseLayer(512, 512),
        Relu(),
        DenseLayer(512, 5)
    ]
    # loss function chosen in the worker()

    timestamp = time.time()
    n_workers = 15  
    """
    valeur à ajuster selon les ressources de votre ordinateur, 
    15 est un bon compromis pour la plupart des machines modernes. 
    Utilisation du parallélisme pour accélérer l'entrainement en traitant plusieurs mini-lots simultanément.
    Pas expliqué dans le rapport, car c'est un sujet complexe propre à la programmation.
    """

    with Pool(n_workers) as p:
        # entrainement par époques
        for epoch in range(num_epochs):
            # mélanger les données
            perm = np.random.permutation(len(input_train))  #aléatoirement mélanger les données d'entrainement
            input_train_shuffled = input_train[perm]
            label_train_shuffled = label_train[perm]

            # entrainement par mini-lots
            for start in range(0, len(input_train), mega_batch_size): #calcul du nombre de mini-lots d'entrainement
                layer_params = get_layer_params(layers)
                end = min(start + mega_batch_size, len(input_train))      
                real_mega_batch_size = end - start
                # entrainement d'un mini-lot
                inputs = input_train_shuffled[start:end]
                labels = label_train_shuffled[start:end]
                x = inputs.reshape(real_mega_batch_size, 1, 32, 32) 
                labels = labels.reshape(real_mega_batch_size, 5) 

                chunk_size = real_mega_batch_size // n_workers + 1
                params = []
                for i in range(0, real_mega_batch_size, chunk_size):
                    j = min(i + chunk_size, real_mega_batch_size)  
                    new_x = x[i:j]
                    new_labels = labels[i:j]
                    params.append((layer_params, new_x, new_labels))

                results = p.starmap(worker, params)
            
                for result in results:
                    add_gradients(layers, result)
                # fin de l'entrainement d'un mini-lot: descente du gradient
                for layer in layers:
                    layer.update(real_mega_batch_size, learning_rate)

                print(f"\rEpoch {epoch+1} | Batch {start//mega_batch_size + 1} processing...", end="")
            
            # tester le modèle
            accuracy = test_model(input_test, label_test, layers)
            print(f"\nEpoch {epoch+1} Done! Test Accuracy: {accuracy*100:.2f}%")
        save_model(layers, accuracy, filename="emoji_cnn_model")

    print(f"finished in {time.time() - timestamp}")