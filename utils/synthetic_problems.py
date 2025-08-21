import torch
from typing import List
import numpy as np

class vlmop2():
    name = "VLMOP2"
    def __init__(self):
        self.upper_bound = 2.0
        self.lower_bound = -2.0
        self.dimension = 4
        self.n_task = 2
    
    def __call__(self, x : torch.Tensor) -> List[torch.Tensor]:
        n = len(x)
        f1 = 1 - torch.exp(-1.0 * torch.sum((x - 1 / np.sqrt(n)) ** 2))
        f2 = 1 - torch.exp(-1.0 * torch.sum((x + 1 / np.sqrt(n)) ** 2))
        return [f1, f2]

class zdt3(): # not used in experiment as algorithm always hit boundaries
    name = "ZDT3"
    def __init__(self):
        self.upper_bound = 1.0
        self.lower_bound = 0.0
        self.dimension = 30
        self.n_task = 2
    
    def __call__(self, x : torch.Tensor) -> List[torch.Tensor]:
        f1 = x[0]
        g = 1 + 9 * torch.mean(x[1:])
        f2 = g * (1 - torch.sqrt(f1 / g) - f1 * torch.sin(10 * np.pi * f1) / g)
        return [f1, f2]

class zdt2(): # not used in experiment as algorithm always hit boundaries
    name = "ZDT2"
    def __init__(self):
        self.upper_bound = 1.0
        self.lower_bound = 0.0
        self.dimension = 30
        self.n_task = 2
    
    def __call__(self, x : torch.Tensor) -> List[torch.Tensor]:
        f1 = x[0]
        g = 1 + 9 * torch.mean(x[1:])
        f2 = g * (1 - torch.pow(f1 / g, 2))
        return [f1, f2]

class omnitest():
    name = "Omnitest"
    def __init__(self):
        self.upper_bound = 6.0
        self.lower_bound = 0.0
        self.dimension = 3
        self.n_task = 2
    
    def __call__(self, x : torch.Tensor) -> List[torch.Tensor]:
        f1 = torch.sum(torch.sin(np.pi * x))
        f2 = torch.sum(torch.cos(np.pi * x))
        return [f1, f2]


problem_dict = {
        "VLMOP2" : vlmop2,
        "Omnitest" : omnitest
    }
