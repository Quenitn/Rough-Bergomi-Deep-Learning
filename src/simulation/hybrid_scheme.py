import numpy as np
from scipy.special import gamma as gamma_fn


class HybridScheme:
    """
    Hybrid scheme de Bennedsen-Lunde-Pakkanen (2017) pour la simulation
    de processus Brownian semistationaires de type Volterra.

    Contrairement a une approximation Riemann pure, on simule exactement
    la loi jointe de (W_t, \\int_0^t (t-s)^{H-1/2} dW_s) sur le premier
    sous-intervalle [t_{i-1}, t_i] via une decomposition de Cholesky 2x2,
    puis on utilise une somme de Riemann pour les pas suivants (k >= 2).
    Cette implementation correspond au schema hybride avec kappa = 1.
    """

    def __init__(self, model, n_steps=100, n_paths=40000):
        self.model = model
        self.n_steps = n_steps
        self.n_paths = n_paths

    @staticmethod
    def _cov_matrix(H, dt):
        """
        Matrice de covariance 2x2 de (W_{t}, Y_t) sur un intervalle de
        longueur dt, ou :
            Y_t = int_0^t (t-s)^{H-1/2} dW_s
        
        On a :
            Var(W_t)      = dt
            Var(Y_t)      = dt^{2H} / (2H)
            Cov(W_t, Y_t) = dt^{H+1/2} / (H + 1/2)
        """
        alpha = H - 0.5
        var_W = dt
        var_Y = dt**(2 * H) / (2 * H)
        cov_WY = dt**(H + 0.5) / (H + 0.5)
        return np.array([[var_W, cov_WY],
                         [cov_WY, var_Y]])

    def _cholesky_params(self, H, dt):
        """
        Retourne les coefficients (a, b, c) tels que :
            W  = a * Z1
            Y  = b * Z1 + c * Z2
        ou (Z1, Z2) sont iid N(0, 1).
        """
        C = self._cov_matrix(H, dt)
        # Cholesky : L @ L.T = C
        L = np.linalg.cholesky(C)
        a = L[0, 0]
        b = L[1, 0]
        c = L[1, 1]
        return a, b, c

    def _riemann_kernel_weights(self, H, dt):
        """
        Poids pour l'approximation Riemann de l'integrale sur les pas k >= 2.
        Formule du papier BLP : pour k >= 2,
            b_k = ((k^{H+1/2} - (k-1)^{H+1/2}) / (H+1/2))^(1/(H-1/2))
        et le poids effectif dans la convolution est alors (b_k * dt)^{H-1/2}.

        Ici on utilise directement la moyenne continue qui correspond au
        choix classique "averaged Riemann" :
            w_k = (k^{H+1/2} - (k-1)^{H+1/2}) / (H + 1/2) * dt^{H-1/2}
        pour k = 2, ..., n_steps.
        """
        alpha = H - 0.5
        k = np.arange(2, self.n_steps + 1)
        w = (k**(alpha + 1) - (k - 1)**(alpha + 1)) / (alpha + 1)
        return w * dt**alpha

    def _build_xi_t(self, xi_0, T_max):
        """Piecewise constant forward variance curve xi_0(t)."""
        maturities = self.model.maturities
        time_grid = np.linspace(0, T_max, self.n_steps)
        indices = np.searchsorted(maturities, time_grid, side='right')
        indices = np.clip(indices, 0, len(xi_0) - 1)
        return xi_0[indices]

    def simulate_paths(self, params):
        """
        Simule les prix S_t et la variance V_t sous rough Bergomi via BLP.
        params : [xi_1, ..., xi_n, nu, rho, H] avec n = len(maturities).
        """
        n_xi = len(self.model.maturities)
        xi_0 = params[:n_xi]
        nu, rho, H = params[n_xi], params[n_xi + 1], params[n_xi + 2]

        T_max = float(self.model.maturities[-1])
        dt = T_max / self.n_steps

        # Cholesky 2x2 de (dW, dY) exact pour le premier pas
        a, b, c = self._cholesky_params(H, dt)

        # Poids Riemann pour les pas suivants (k >= 2)
        riemann_w = self._riemann_kernel_weights(H, dt)

        # Variables gaussiennes independantes
        # Z1 drives dW, Z2 is the independent component of Y
        Z1 = np.random.standard_normal((self.n_paths, self.n_steps))
        Z2 = np.random.standard_normal((self.n_paths, self.n_steps))
        # Z_perp for independent Brownian driving the asset
        Z_perp = np.random.standard_normal((self.n_paths, self.n_steps))

        # Increments du Brownien principal : dW_i = a * Z1_i = sqrt(dt) * Z1_i
        dW1 = a * Z1  # shape (n_paths, n_steps)

        # "Kernel-weighted" contribution on each interval : dY_i (exact part)
        # Y_i = b * Z1_i + c * Z2_i
        dY_exact = b * Z1 + c * Z2

        # On construit W^H_t pour chaque t en combinant :
        #  - la contribution exacte (BLP) sur le dernier pas
        #  - la contribution Riemann (approximee) sur les pas k >= 2
        vol_process = np.zeros((self.n_paths, self.n_steps))

        # Pour t = 0, vol_process[:, 0] = 0 (rien ne s'est passe)
        # Pour t = 1 (premier pas simule), seule la partie exacte joue
        # Pour t >= 2, on somme : partie exacte au dernier pas + Riemann sur les dW1 anterieurs
        for t in range(1, self.n_steps):
            # Partie exacte : contribution du pas le plus recent
            vol_process[:, t] = dY_exact[:, t - 1]
            # Partie Riemann : sum_{k=2}^{t} riemann_w[k-2] * dW1[t - k]
            if t >= 2:
                # riemann_w est indexe pour k = 2, 3, ...
                # on veut sum_{k=2}^{t} riemann_w[k-2] * dW1[:, t-k]
                # c'est une convolution : vol_process[:, t] += dot(dW1[:, :t-1], riemann_w[:t-1][::-1])
                n_terms = t - 1  # k va de 2 a t, soit t-1 termes
                vol_process[:, t] += np.dot(
                    dW1[:, :n_terms],
                    riemann_w[:n_terms][::-1]
                )

        # Brownien correle pour le spot
        # dW2 = rho * dW1 + sqrt(1-rho^2) * dW_perp
        dW_perp = np.sqrt(dt) * Z_perp
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp

        # Variance V_t
        xi_t = self._build_xi_t(xi_0, T_max)
        time_grid = np.linspace(0, T_max, self.n_steps)
        V = xi_t[np.newaxis, :] * np.exp(
            nu * np.sqrt(2 * H) * vol_process
            - 0.5 * (nu**2) * np.power(time_grid, 2 * H)
        )

        # Prix log-normal
        log_S = np.zeros((self.n_paths, self.n_steps))
        for t in range(1, self.n_steps):
            log_S[:, t] = (log_S[:, t - 1]
                            - 0.5 * V[:, t - 1] * dt
                            + np.sqrt(V[:, t - 1]) * dW2[:, t - 1])

        return np.exp(log_S), V