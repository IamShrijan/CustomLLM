import torch.nn as nn 
import torch 
from GELU import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super.__init__()
        self.layer1 = nn.Linear(cfg["emd_dim"], 4* cfg["emd_dim"])
        self.layer2 = nn.Linear(4 * cfg["emd_dim"],cfg["emd_dim"])
    def forward(self, x):
        x = self.layer1(x)
        x = GELU(x)
        x = self.layer2(x)
        return x