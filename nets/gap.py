import torch
import torch.nn as nn

class GAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        # [b, c, h, w] -> [b, c]
        return torch.mean(x, dim=[2,3])