import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAutoencoder, self).__init__()
        # Encoder: npr. 30 -> 16 -> 8 -> 4 (Latentni prostor)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4) 
        )
        # Decoder: 4 -> 8 -> 16 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim) # Izlaz je iste dimenzije kao ulaz
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
