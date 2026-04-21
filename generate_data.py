"""
Script CLI pour generer le dataset baseline. Appelle le pipeline
centralise dans src/utils/pipeline.py.

Usage : python generate_data.py
"""

import os
import numpy as np

from src.model.rough_bergomi import RoughBergomi
from src.utils.pipeline import generate_dataset
from src.utils.normalization import normalize_inputs, normalize_outputs

# --- CONFIGURATION ---
N_SAMPLES = 1000    # Nombre de surfaces a generer (test rapide)
N_PATHS = 10000     # Chemins Monte Carlo par surface
N_STEPS = 100       # Pas de discretisation temporelle
SAVE_PATH = "data/"

# Valeurs recommandees pour le run final :
# N_SAMPLES = 20000
# N_PATHS = 30000


def generate():
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"--- Generation du dataset baseline ---")
    print(f"N_SAMPLES = {N_SAMPLES}, N_PATHS = {N_PATHS}, N_STEPS = {N_STEPS}")

    model = RoughBergomi()
    X, Y = generate_dataset(model, N_SAMPLES, n_paths=N_PATHS, n_steps=N_STEPS)

    # Sauvegarde des donnees brutes
    np.save(os.path.join(SAVE_PATH, "X_params_raw.npy"), X)
    np.save(os.path.join(SAVE_PATH, "Y_vols_raw.npy"), Y)

    # Normalisation + sauvegarde
    X_norm = normalize_inputs(X, model.bounds)
    Y_norm, y_mean, y_std = normalize_outputs(Y, per_point=True)

    np.save(os.path.join(SAVE_PATH, "X_params_norm.npy"), X_norm)
    np.save(os.path.join(SAVE_PATH, "Y_vols_norm.npy"), Y_norm)
    np.savez(os.path.join(SAVE_PATH, "norm_stats.npz"),
             y_mean=y_mean, y_std=y_std)

    print("Fichiers sauvegardes dans data/. Pret pour l'entrainement.")


if __name__ == "__main__":
    generate()
