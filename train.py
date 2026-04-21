"""
Script CLI pour entrainer le reseau sur le dataset baseline.
Delegue la boucle d'entrainement a src/nn/training.py.

Usage : python train.py
"""

import os
import numpy as np
import torch

from src.nn.training import train_model

# --- CONFIGURATION ---
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def train():
    if not os.path.exists("data/X_params_norm.npy"):
        print("Erreur : lance d'abord python generate_data.py")
        return

    print("--- Chargement des donnees ---")
    X = np.load("data/X_params_norm.npy").astype(np.float32)
    Y = np.load("data/Y_vols_norm.npy").astype(np.float32)
    print(f"Dataset : {X.shape[0]} surfaces, input dim {X.shape[1]}, output dim {Y.shape[1]}")

    print(f"\n--- Entrainement ---")
    net, history = train_model(
        X, Y,
        epochs=EPOCHS, patience=PATIENCE,
        batch=BATCH_SIZE, lr=LEARNING_RATE,
        verbose=True
    )

    # Sauvegarde
    torch.save(net.state_dict(), "data/model_weights.pth")
    np.savez("data/training_history.npz",
             train=np.array(history['train']),
             val=np.array(history['val']))

    print("Modele et historique sauvegardes.")


if __name__ == "__main__":
    train()
