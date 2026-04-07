import numpy as np
import os
from src.model.rough_bergomi import RoughBergomi
from src.simulation.hybrid_scheme import HybridScheme
from src.utils.black_scholes import implied_vol

N_SAMPLES = 100 # Teste avec 100 d'abord
N_PATHS = 30000 # On monte un peu en précision
SAVE_PATH = "data/"

def generate():
    model = RoughBergomi()
    # On définit les grilles depuis le modèle
    maturities = model.maturities # [0.1, 0.3, ..., 2.0] (8 points)
    strikes = model.strikes       # [0.5, ..., 1.5] (11 points)
    
    simulator = HybridScheme(model, n_paths=N_PATHS)
    X = model.sample_parameters(n_samples=N_SAMPLES)
    Y = np.zeros((N_SAMPLES, len(maturities) * len(strikes)))

    print("--- Début de la génération réelle ---")

    for i in range(N_SAMPLES):
        # 1. Simuler les chemins pour ce jeu de paramètres
        prices, _ = simulator.simulate_paths(X[i])
        
        # 2. Calculer la nappe de vol pour chaque couple (T, K)
        vol_surface = []
        for t_idx, T in enumerate(maturities):
            # On récupère les prix à l'instant T (la colonne correspondante)
            # Attention : il faut mapper les pas de temps du simulateur aux maturités
            step_idx = int(T * (simulator.n_steps / maturities[-1])) - 1
            S_at_T = prices[:, step_idx]
            S0 = 1.0 # Par convention dans ton modèle
            
            for K in strikes:
                # Prix de l'option Call (Moyenne des payoffs)
                payoff = np.maximum(S_at_T - K, 0)
                mkt_price = np.mean(payoff)
                
                # Conversion en Vol Implicite
                iv = implied_vol(mkt_price, S0, K, T, r=0.0)
                vol_surface.append(iv)
        
        Y[i, :] = np.array(vol_surface)
        
        if i % 5 == 0:
            print(f"Échantillon {i}/{N_SAMPLES} généré...")

    # Sauvegarde
    np.save(os.path.join(SAVE_PATH, "X_params.npy"), X)
    np.save(os.path.join(SAVE_PATH, "Y_vols.npy"), Y)
    print("Fichiers sauvegardés. Prêt pour le Deep Learning !")

if __name__ == "__main__":
    generate()