import numpy as np

class BaseModel:
    def __init__(self, maturities=None, strikes=None):
        # 8 Maturités (T) du papier par défaut (Section 4.1.1)
        if maturities is not None:
            self.maturities = np.asarray(maturities, dtype=np.float64)
        else:
            self.maturities = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
        
        # 11 Strikes (k) du papier par défaut (Section 4.1.1)
        if strikes is not None:
            self.strikes = np.asarray(strikes, dtype=np.float64)
        else:
            self.strikes = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        
        # Dimension de sortie : calculée dynamiquement
        self.n_output = len(self.maturities) * len(self.strikes)