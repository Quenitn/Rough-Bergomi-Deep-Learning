import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.nn.model import VolatilityNet
import os

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Optimisation pour Mac M1/M2/M3

def train():
    print(f"--- Entraînement sur {DEVICE} ---")

    # 1. Chargement des données (npy -> torch tensors)
    if not os.path.exists("data/X_params.npy"):
        print("Erreur : Génère des données d'abord !")
        return

    x_data = np.load("data/X_params.npy").astype(np.float32)
    y_data = np.load("data/Y_vols.npy").astype(np.float32)

    # Conversion en Tenseurs PyTorch
    X = torch.from_numpy(x_data)
    Y = torch.from_numpy(y_data)

    # 2. Création du DataLoader (pour donner à manger au réseau par petits lots)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialisation du modèle, de la fonction de perte et de l'optimiseur
    model = VolatilityNet().to(DEVICE)
    criterion = nn.MSELoss() # Mean Squared Error : on veut minimiser l'écart au carré
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Boucle d'entraînement
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Forward pass (Le fameux bouton ON !)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass (L'apprentissage : on corrige les erreurs)
            optimizer.zero_grad() # On remet le compteur à zéro
            loss.backward()       # On calcule l'erreur pour chaque neurone
            optimizer.step()      # On ajuste les poids
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # 5. Sauvegarde du cerveau entraîné
    torch.save(model.state_dict(), "data/model_weights.pth")
    print("--- Entraînement terminé et modèle sauvegardé ! ---")

if __name__ == "__main__":
    train()