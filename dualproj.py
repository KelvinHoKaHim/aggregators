from typing import Callable, List
import torch
import torch.autograd.functional as F
from torch.autograd import grad 
import cvxpy as cp
from cvxopt import matrix, solvers
import numpy as np
import numpy

def DualProj(jacobian : torch.Tensor) -> (torch.Tensor, numpy.float64):
    # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
    # Output : descent direction of length d, and alpha of length n.
    jac = jacobian.detach().cpu().numpy()
    m = len(jac)
    G = jac @ jac.T 

    G_norm = np.sum(np.diag(G)) #normalization to ensure positive definiteness and full rank
    if G_norm < 1e-4:
        G = np.zeros_like(G)
    G = G / G_norm
    G = G + 1e-4 * np.eye(len(G))

    P = matrix(G)                           # Minimize v^T JJ^T v
    q = matrix(np.zeros(m))                 # with constraint
    G_cvx = matrix(-1.0 * np.eye(m))        # v_i >= 1 for each i
    h = matrix((-1.0 / m) * np.ones(m))     
    A = None 
    b = None
    sol = solvers.qp(P, q, G_cvx, h, A, b)
    alpha = np.array(sol['x']).squeeze()

    #alpha = alpha / np.sum(alpha)  # Normalize alpha to unit length
    direction = jac.T @ alpha
    direction = torch.Tensor(direction)
    return (direction, alpha)


