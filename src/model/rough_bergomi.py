import numpy as np
from src.model.base import BaseModel

class RoughBergomi(BaseModel):
    def __init__(self):
        # Initialise la grille (8 maturités, 11 strikes) de la classe parente
        super().__init__()
        
        # Bornes des paramètres (Domaines d'admissibilité) - Page 20 du papier 
        # xi_0 : Forward variance (courbe initiale)
        # nu : Volatilité de la volatilité (eta dans certaines notations)
        # rho : Corrélation entre l'actif et la volatilité
        # H : Paramètre de Hurst (rugosité)
        
        self.bounds = {
            'xi': (0.01, 0.16),    # Pour les 8 points de la courbe 
            'nu': (0.5, 4.0),     # 
            'rho': (-0.95, -0.1),  # 
            'H': (0.025, 0.5)      # 
        }

    def sample_parameters(self, n_samples):
        """
        Génère n_samples combinaisons aléatoires de paramètres theta.
        Chaque theta est un vecteur de taille 11 : [xi1...xi8, nu, rho, H]
        """
        # Tirage uniforme pour les 8 points de xi_0
        xi_samples = np.random.uniform(self.bounds['xi'][0], self.bounds['xi'][1], (n_samples, 8))
        
        # Tirage uniforme pour nu, rho et H
        nu_samples = np.random.uniform(self.bounds['nu'][0], self.bounds['nu'][1], (n_samples, 1))
        rho_samples = np.random.uniform(self.bounds['rho'][0], self.bounds['rho'][1], (n_samples, 1))
        h_samples = np.random.uniform(self.bounds['H'][0], self.bounds['H'][1], (n_samples, 1))
        
        # On concatène tout pour obtenir une matrice (n_samples, 11) 
        return np.hstack([xi_samples, nu_samples, rho_samples, h_samples])