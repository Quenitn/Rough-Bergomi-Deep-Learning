import numpy as np

class BaseModel:
    def __init__(self):
        # 8 Maturités (T) du papier (Section 4.1.1)
        self.maturities = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]) 
        
        # 11 Strikes (k) du papier (Section 4.1.1)
        self.strikes = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]) 
        
        # Dimension de sortie : 88 points [cite: 307, 338]
        self.n_output = len(self.maturities) * len(self.strikes)