"""
Pipeline de generation de donnees : de la simulation Monte Carlo
a la surface de volatilite implicite, et aggregation en dataset.
Mutualise le code utilise par generate_data.py et par le notebook
pour les experiences 6 et 7.
"""

import numpy as np
from src.simulation.hybrid_scheme import HybridScheme
from src.utils.black_scholes import implied_vol


def compute_surface_from_prices(prices, maturities, strikes, T_max, n_steps, S0=1.0):
    dt = T_max / n_steps
    surface = np.zeros((len(maturities), len(strikes)))

    for i, T in enumerate(maturities):
        step_idx = int(round(T / dt)) - 1
        step_idx = max(0, min(step_idx, n_steps - 1))
        S_T = prices[:, step_idx]
        
        # Control variate : force martingalité E[S_T] = S0
        S_T = S_T * S0 / S_T.mean()

        for j, K in enumerate(strikes):
            payoff = np.maximum(S_T - K, 0.0)
            price_mc = payoff.mean()
            surface[i, j] = implied_vol(price_mc, S0, K, T)

    return surface


def generate_dataset(model, n_samples, n_paths=30000, n_steps=100,
                     verbose=True, print_every=None, seed=None):
    """
    Genere un dataset (X, Y) pour un modele rough Bergomi donne.

    Parameters
    ----------
    model : RoughBergomi
        Modele configure avec ses bounds, maturites et strikes.
    n_samples : int
        Nombre de surfaces a generer.
    n_paths : int
        Chemins Monte Carlo par surface.
    n_steps : int
        Pas de discretisation.
    verbose : bool
    print_every : int, optional
        Frequence de l'affichage. Par defaut, environ 20 messages au total.
    seed : int, optional
        Pour reproductibilite de sample_parameters.

    Returns
    -------
    X : ndarray, shape (n_samples, n_maturities + 3)
    Y : ndarray, shape (n_samples, n_maturities * n_strikes)
    """
    if seed is not None:
        np.random.seed(seed)

    simulator = HybridScheme(model, n_steps=n_steps, n_paths=n_paths)
    maturities = model.maturities
    strikes = model.strikes
    T_max = float(maturities[-1])

    X = model.sample_parameters(n_samples)
    Y = np.zeros((n_samples, len(maturities) * len(strikes)))

    if print_every is None:
        print_every = max(1, n_samples // 20)

    for i in range(n_samples):
        prices, _ = simulator.simulate_paths(X[i])
        surface = compute_surface_from_prices(
            prices, maturities, strikes, T_max, simulator.n_steps
        )
        Y[i] = surface.flatten()

        if verbose and (i + 1) % print_every == 0:
            print(f"  {i+1}/{n_samples}")
    
    # Filtrer les surfaces contenant des NaN ou positifs 
    # Garde les surfaces avec au plus 20% de NaN
    max_nans = int(0.2 * Y.shape[1])
    valid = np.isnan(Y).sum(axis=1) <= max_nans
    n_dropped = (~valid).sum()
    if n_dropped > 0 and verbose:
        print(f"  Dropped {n_dropped} surfaces with invalid implied vols "
              f"({n_dropped}/{n_samples} = {100*n_dropped/n_samples:.1f}%)")
    X = X[valid]
    Y = Y[valid]


    
    
    return X, Y
