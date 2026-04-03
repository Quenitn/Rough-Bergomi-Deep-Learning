import numpy as np
import os
from src.model.rough_bergomi import RoughBergomi
from src.simulation.hybrid_scheme import HybridScheme
from src.utils.black_scholes import implied_vol

# --- CONFIGURATION ---
N_SAMPLES = 100  # On commence petit (change à 68000 plus tard)
N_PATHS = 10000  # Précision pour chaque point
SAVE_PATH = "data/"

def generate():
    print(f"--- Lancement de la génération ({N_SAMPLES} échantillons) ---")
    
    model = RoughBergomi()
    simulator = HybridScheme(model, n_paths=N_PATHS)
    
    # 1. Tirage des paramètres (X)
    X = model.sample_parameters(n_samples=N_SAMPLES)
    
    # 2. Préparation du tableau de sortie (Y) 
    # 8 maturités * 11 strikes = 88 points
    Y = np.zeros((N_SAMPLES, 88))
    
    for i in range(N_SAMPLES):
        if i % 10 == 0:
            print(f"Progression : {i}/{N_SAMPLES}...")
            
        # Simulation des prix pour ce jeu de paramètres
        prices, _ = simulator.simulate_paths(X[i])
        
        # Calcul du prix moyen final (Monte Carlo)
        final_prices = np.mean(prices[:, -1]) # Simplifié pour le test
        
        # TODO: Ici on calculera les 88 points avec une boucle sur la grille
        # Pour l'instant on met une valeur fictive pour tester la sauvegarde
        Y[i, :] = np.random.random(88) 

    # 3. Sauvegarde
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    np.save(os.path.join(SAVE_PATH, "X_params.npy"), X)
    np.save(os.path.join(SAVE_PATH, "Y_vols.npy"), Y)
    
    print(f"--- Terminé ! Fichiers sauvegardés dans {SAVE_PATH} ---")

if __name__ == "__main__":
    generate()