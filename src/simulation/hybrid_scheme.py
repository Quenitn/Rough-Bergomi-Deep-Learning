import numpy as np
from src.utils.black_scholes import implied_vol

class HybridScheme:
    def __init__(self, model, n_steps=100, n_paths=40000):
        self.model = model
        self.n_steps = n_steps   # Précision du temps (100 pas par an)
        self.n_paths = n_paths   # 40 000 scénarios pour la moyenne
        
    def get_kernel_weights(self, H, dt):
        """Calcule les poids de l'intégrale de Volterra (Riemann-Liouville)."""
        alpha = H - 0.5
        # b_k sont les poids qui donnent l'aspect 'rugueux' (Page 15 du papier)
        k = np.arange(1, self.n_steps + 1)
        weights = (np.power(k, alpha + 1) - np.power(k - 1, alpha + 1)) / (alpha + 1)
        return weights * np.power(dt, alpha)

    def simulate_paths(self, params):
        """
        Simule les prix S_t et la variance V_t.
        params : [xi1...8, nu, rho, H]
        """
        # 1. On sépare les 11 paramètres
        xi_0 = params[:8]  # Courbe de variance initiale
        nu, rho, H = params[8], params[9], params[10]
        
        T_max = 2.0
        dt = T_max / self.n_steps
        
        # 2. Génération des bruits (Mouvements Browniens)
        # dW1 pour la Vol, dW_perp pour l'indépendance
        dW1 = np.random.normal(0, np.sqrt(dt), (self.n_paths, self.n_steps))
        dW_perp = np.random.normal(0, np.sqrt(dt), (self.n_paths, self.n_steps))
        # dW2 est le bruit du prix, corrélé à rho avec dW1
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp
        
        # 3. Calcul de la Volatilité (Noyau de Volterra)
        kernel = self.get_kernel_weights(H, dt)
        vol_process = np.zeros((self.n_paths, self.n_steps))
        
        # Convolution pour l'intégrale stochastique
        for t in range(1, self.n_steps):
            vol_process[:, t] = np.dot(dW1[:, :t], kernel[:t][::-1])
            
        # Variance V_t (Modèle Log-Normal de Bergomi)
        # On utilise la moyenne de xi_0 pour simplifier le test
        V = np.mean(xi_0) * np.exp(nu * np.sqrt(2 * H) * vol_process - 0.5 * (nu**2) * np.power(np.linspace(0, T_max, self.n_steps), 2 * H))
        
        # 4. Calcul du Prix S_t
        log_S = np.zeros((self.n_paths, self.n_steps))
        for t in range(1, self.n_steps):
            log_S[:, t] = log_S[:, t-1] - 0.5 * V[:, t-1] * dt + np.sqrt(V[:, t-1]) * dW2[:, t-1]
            
        return np.exp(log_S), V # Retourne les trajectoires des prix