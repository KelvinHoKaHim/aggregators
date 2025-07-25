import torch
from typing import List, Callable, Any
import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers


'''def MGDA(jacobian : torch.Tensor, n_iter : int = 100) -> torch.Tensor:
        m = len(jacobian)
        G = (jacobian @ jacobian.t()).cpu().detach().numpy()
        G_norm = np.sum(np.diag(G))
        if G_norm < 1e-4:
            G = np.zeros_like(G)
        #G = G / G_norm
        G = G + 1e-4 * np.eye(len(G))
        Q = G
        Q = matrix(np.float64(Q))
        p = np.zeros(m)
        A = np.ones(m)

        A = matrix(A, (1, m))
        b = matrix(1.0)

        G_cvx = -np.eye(m)
        h = [0.0] * m
        h = matrix(h)

        G_cvx = matrix(G_cvx)
        p = matrix(p)
        sol = solvers.qp(Q, p, G_cvx, h, A, b)

        res = np.array(sol['x']).squeeze()
        alpha = res / sum(res)  # important. Does res already satisfy sum=1?
        alpha_torch = torch.Tensor(alpha)
        descent_direction = jacobian.t() @ alpha_torch
        return (descent_direction, alpha)'''

def MGDA(jacobian : torch.Tensor, max_iters : int = 100, epsilon: float = 1e-4) -> Callable[[torch.Tensor], tuple[torch.Tensor, np.ndarray]]:
    # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
    # Output : descent direction of length d, and alpha of length n.
    gramian = jacobian @ jacobian.t() 
    device = gramian.device
    dtype = gramian.dtype
    alpha = torch.ones(gramian.shape[0], device=device, dtype=dtype) / gramian.shape[0]

    for i in range(max_iters): # Iterate Frank-Wolfe several times.
        t = torch.argmin(gramian @ alpha)
        e_t = torch.zeros(gramian.shape[0],  device=device, dtype=dtype)
        e_t[t] = 1.0
        a = alpha @ (gramian @ e_t)
        b = alpha @ (gramian @ alpha)
        c = e_t @ (gramian @ e_t)
        if c <= a:
            gamma = 1.0
        elif b <= a:
            gamma = 0.0
        else:
            gamma = (b - a) / (b + c - 2 * a)  
        alpha = (1 - gamma) * alpha + gamma * e_t
        if gamma < epsilon:
            break
        
    alpha = alpha / torch.sum(alpha)  # Normalize alpha to unit length
    descent_direction = jacobian.t() @ alpha
    alpha = alpha.detach().cpu().numpy()
    return (descent_direction, alpha)