import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, sigma, r=0.0, option='call'):
    # Sécurité pour éviter division par zéro
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option == 'call' else max(K - S, 0.0)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r=0.0):
    intrinsic_value = max(S - K, 0.0)
    if price <= intrinsic_value + 1e-7:
        return np.nan          
    try:
        f = lambda x: bs_price(S, K, T, x, r) - price
        return brentq(f, 1e-6, 5.0)
    except (ValueError, RuntimeError):
        return np.nan         