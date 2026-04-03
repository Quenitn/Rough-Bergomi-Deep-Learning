from src.model.rough_bergomi import RoughBergomi
from src.simulation.hybrid_scheme import HybridScheme
import numpy as np

# 1. On prépare le modèle
print("--- DEBUT DU TEST ---")
model = RoughBergomi()

# 2. On prépare le simulateur (on met peu de paths pour que ce soit rapide)
simulator = HybridScheme(model, n_steps=50, n_paths=5000)

# 3. On tire UN SEUL scénario (11 paramètres)
params = model.sample_parameters(n_samples=1)[0]
print(f"Paramètres tirés (H, rho, nu...) : {params[-3:]}")

# 4. On lance la simulation
prices,vol = simulator.simulate_paths(params)

print(f"Simulation terminée !")
print(f"Forme de la matrice de prix : {prices.shape}") # Doit être (5000, 50)