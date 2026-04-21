import numpy as np
from src.model.base import BaseModel

class RoughBergomi(BaseModel):
    def __init__(self, maturities=None, strikes=None):
        # Initialise la grille (maturités, strikes) de la classe parente
        super().__init__(maturities=maturities, strikes=strikes)
        
        # Bornes des paramètres (Domaines d'admissibilité) - Page 20 du papier 
        # xi_0 : Forward variance (courbe initiale)
        # nu : Volatilité de la volatilité (eta dans certaines notations)
        # rho : Corrélation entre l'actif et la volatilité
        # H : Paramètre de Hurst (rugosité)
        
        self.bounds = {
            'xi': (0.01, 0.16),    # Pour les 8 points de la courbe 
            'nu': (0.5, 4.0),
            'rho': (-0.95, -0.1),
            'H': (0.025, 0.5)
        }

    @classmethod
    def market_regime(cls, regime, maturities=None, strikes=None):
        """
        Crée une instance RoughBergomi avec des bounds adaptés à un régime de marché.
        
        Régimes disponibles :
        - "equity_index" : SPX-like (bounds par défaut du papier)
        - "high_vol"     : crypto, commodités (ξ₀ élevé, ν large)
        - "low_vol_fx"   : FX majors (ξ₀ faible, corrélation symétrique)
        """
        instance = cls(maturities=maturities, strikes=strikes)
        
        regimes = {
            'equity_index': {
                'xi': (0.01, 0.16),
                'nu': (0.5, 4.0),
                'rho': (-0.95, -0.1),
                'H': (0.025, 0.5),
            },
            'high_vol': {
                'xi': (0.05, 0.50),
                'nu': (1.0, 6.0),
                'rho': (-0.7, 0.1),
                'H': (0.05, 0.3),
            },
            'low_vol_fx': {
                'xi': (0.005, 0.04),
                'nu': (0.3, 2.0),
                'rho': (-0.4, 0.4),
                'H': (0.05, 0.4),
            },
        }
        
        if regime not in regimes:
            raise ValueError(f"Régime inconnu '{regime}'. Choix : {list(regimes.keys())}")
        
        instance.bounds = regimes[regime]
        return instance

    def sample_parameters(self, n_samples):
        """
        Génère n_samples combinaisons aléatoires de paramètres theta.
        Chaque theta est un vecteur de taille 11 : [xi1...xi8, nu, rho, H]
        """
        n_xi = len(self.maturities)  # Nombre de points de la courbe ξ₀
        
        # Tirage uniforme pour les points de xi_0
        xi_samples = np.random.uniform(
            self.bounds['xi'][0], self.bounds['xi'][1], (n_samples, n_xi)
        )
        
        # Tirage uniforme pour nu, rho et H
        nu_samples = np.random.uniform(self.bounds['nu'][0], self.bounds['nu'][1], (n_samples, 1))
        rho_samples = np.random.uniform(self.bounds['rho'][0], self.bounds['rho'][1], (n_samples, 1))
        h_samples = np.random.uniform(self.bounds['H'][0], self.bounds['H'][1], (n_samples, 1))
        
        # On concatène tout pour obtenir une matrice (n_samples, n_xi + 3) 
        return np.hstack([xi_samples, nu_samples, rho_samples, h_samples])