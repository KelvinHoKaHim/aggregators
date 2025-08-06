from typing import Callable, List
import torch
import torch.autograd.functional as F
from torch.autograd import grad 
import cvxpy as cp
from cvxopt import matrix, solvers
import numpy as np
import numpy

def UPGrad(jacobian : torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
    # Output : descent direction of length d, and alpha of length n.
    m = len(jac)
    G = jac @ jac.T

    G_norm = np.trace(G)
    if G_norm < 1e-4:
        G = np.zeros_like(G) # If the norm is too small, we set it to zero
    else:
        G = G / G_norm  # Normalize the Gramian matrix, avoid numerical issues when entries of G is too small
    G = G + 1e-4 * np.eye(len(G)) # to ensure positive definiteness

    # UPgrad projects each gradient to dual cone of the Jacobian, to get projected gradients
    # You average the projected gradients (implementation-wise, you may be averaging the alphas)
    # That projection will be in the cone as well (alphas are non-negative)

    P = matrix(np.float64(G))
    q = matrix(np.zeros(m))
    A = None
    b = None
    G_cvx = matrix(-1.0*np.eye(m))
    alpha = np.zeros(m)
    for i in range(m): # project each gradient to the dual cone, then add up the projected gradients
        neg_e_i = np.zeros(m)  
        neg_e_i[i] = -1.0 / m                   # Minimize v^T JJ^T v 
        h = matrix(neg_e_i)                     # Subject to v \preceq e_i / n
        sol = solvers.qp(P, q, G_cvx, h, A, b)  # UPGrad is linear under scaling
        w = np.array(sol['x']).squeeze()
        alpha = alpha + w

    #alpha = alpha / np.sum(alpha)  # Normalize alpha 
    direction = jac.T @ alpha
    direction = torch.Tensor(direction)
    return (direction, alpha)


