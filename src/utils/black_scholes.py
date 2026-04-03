import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, sigma, r=0.0, option='call'):
    """Calcule le prix théorique d'une option via Black-Scholes[cite: 269]."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) [cite: 270]
    d2 = d1 - sigma * np.sqrt(T) [cite: 270]
    
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) [cite: 269]
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r=0.0):
    """Retrouve la volatilité sigma via le solveur de Brent[cite: 271, 272]."""
    if price <= 0: return 0.0
    try:
        # On cherche sigma tel que BS(sigma) - price_market = 0
        f = lambda x: bs_price(S, K, T, x, r) - price
        return brentq(f, 1e-6, 5.0) # On cherche entre 0.0001% et 500%
    except:
        return 0.0