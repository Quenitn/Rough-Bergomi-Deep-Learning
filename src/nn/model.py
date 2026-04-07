import torch
import torch.nn as nn

class VolatilityNet(nn.Module):
    def __init__(self, input_size=11, output_size=88):
        super(VolatilityNet, self).__init__()
        
        # Architecture : 11 -> 512 -> 512 -> 512 -> 512 -> 88
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ELU(),  # ELU est souvent mieux que ReLU pour les surfaces de vol
            
            nn.Linear(512, 512),
            nn.ELU(),
            
            nn.Linear(512, 512),
            nn.ELU(),
            
            nn.Linear(512, 512),
            nn.ELU(),
            
            nn.Linear(512, output_size), # Sortie directe des 88 points
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)