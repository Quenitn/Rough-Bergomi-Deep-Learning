import torch
import torch.nn as nn

class VolatilityNet(nn.Module):
    def __init__(self, input_size=11, output_size=88):
        super(VolatilityNet, self).__init__()
        
        # Architecture fidèle au papier (Section 3.2.1) :
        # 11 -> 30 -> 30 -> 30 -> 30 -> 88 (~6800 paramètres)
        # Activation ELU sur les couches cachées, linéaire en sortie (Section 4.1.1)
        self.network = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.ELU(),
            
            nn.Linear(30, 30),
            nn.ELU(),
            
            nn.Linear(30, 30),
            nn.ELU(),
            
            nn.Linear(30, 30),
            nn.ELU(),
            
            nn.Linear(30, output_size),  # Sortie linéaire (pas de ReLU)
        )

    def forward(self, x):
        return self.network(x)