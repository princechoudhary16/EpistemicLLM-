import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        return torch.sigmoid(self.linear(self.norm(hidden.float()))).squeeze(-1)
