"""
Normalisation des entrees/sorties pour l'entrainement du reseau.
Separee de generate_data.py pour etre reutilisable dans le notebook
et dans les experiences de changement de grille / de regime.
"""

import numpy as np


def normalize_inputs(X, bounds, n_xi=None):
    """
    Normalisation min-max des parametres vers [-1, 1].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_xi + 3)
        Parametres bruts. Les 3 dernieres colonnes sont (nu, rho, H).
    bounds : dict
        Clefs 'xi', 'nu', 'rho', 'H', chacune mappant un couple (lo, hi).
    n_xi : int, optional
        Nombre de composantes xi. Si None, deduit de X.shape[1] - 3.

    Returns
    -------
    X_norm : ndarray, meme shape que X
    """
    if n_xi is None:
        n_xi = X.shape[1] - 3

    col_min = np.zeros(X.shape[1])
    col_max = np.zeros(X.shape[1])
    col_min[:n_xi] = bounds['xi'][0]
    col_max[:n_xi] = bounds['xi'][1]
    col_min[n_xi]     = bounds['nu'][0]
    col_max[n_xi]     = bounds['nu'][1]
    col_min[n_xi + 1] = bounds['rho'][0]
    col_max[n_xi + 1] = bounds['rho'][1]
    col_min[n_xi + 2] = bounds['H'][0]
    col_max[n_xi + 2] = bounds['H'][1]

    return 2.0 * (X - col_min) / (col_max - col_min) - 1.0


def normalize_outputs(Y, per_point=True):
    """
    Normalisation des surfaces par moyenne / ecart-type.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_grid_points)
    per_point : bool
        Si True, normalisation point-par-point (vecteur de mean / std).
        Si False, normalisation scalaire globale. Le papier utilise
        une normalisation par point, ce qui accelere la convergence.

    Returns
    -------
    Y_norm : ndarray
    y_mean : ndarray ou scalaire
    y_std : ndarray ou scalaire
    """
    if per_point:
        y_mean = np.nanmean(Y, axis=0)
        y_std = np.nanstd(Y, axis=0) + 1e-12
    else:
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-12:
            y_std = 1.0

    Y_norm = (Y - y_mean) / y_std
    return Y_norm, y_mean, y_std


def denormalize_outputs(Y_norm, y_mean, y_std):
    """Inverse de normalize_outputs."""
    return Y_norm * y_std + y_mean
