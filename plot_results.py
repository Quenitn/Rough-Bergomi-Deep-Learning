import torch
import numpy as np
import matplotlib.pyplot as plt
from src.nn.model import VolatilityNet

def plot_comparison():
    # 1. Charger le modèle et les données
    model = VolatilityNet()
    model.load_state_dict(torch.load("data/model_weights.pth"))
    model.eval() # Mode évaluation (pas d'apprentissage)

    X = np.load("data/X_params.npy").astype(np.float32)
    Y_true = np.load("data/Y_vols.npy").astype(np.float32)

    # 2. Prendre un échantillon au hasard (ex: le premier)
    idx = 0 
    sample_x = torch.from_numpy(X[idx:idx+1])
    
    with torch.no_grad():
        Y_pred = model(sample_x).numpy().flatten()
    
    Y_true_sample = Y_true[idx]

    # 3. Graphique : Comparaison des Smiles de Volatilité
    # On va afficher la première maturité (les 11 premiers points)
    strikes = np.linspace(0.5, 1.5, 11)
    
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, Y_true_sample[:11], 'ok', label="Simulation (Vérité)", markersize=8)
    plt.plot(strikes, Y_pred[:11], '-r', label="IA (Prédiction)", linewidth=2)
    
    plt.title(f"Comparaison du Smile de Volatilité (Maturité T=0.1)")
    plt.xlabel("Strike (K)")
    plt.ylabel("Volatilité Implicite")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_comparison()