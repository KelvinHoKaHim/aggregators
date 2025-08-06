import torch
from typing import List, Callable, Any
import numpy as np
import cvxpy as cp
import sys 
import numpy

#input jacobian , output descent direction

def NASH_MTL(jacobian : torch.Tensor, n_iter : int = 100) -> (torch.Tensor, numpy.float64):
    # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
    # Output : descent direction of length d, and alpha of length n.
    # alpha has non-negative values as constraint
    # Using formulation 5 in Navon et. al., with lixfnerization as Navon et. al. recommended
    # See documentation for details of this implementation
    n_task = len(jacobian)
    G = jacobian.detach().t().cpu().numpy()

    GTG = G.T @ G
    GTG_normalise = np.linalg.norm(GTG)#cp.Parameter(shape = 1, value = np.linalg.norm(GTG)) # normalisation
    GTG_param = cp.Parameter(shape = (n_task, n_task), value = GTG / GTG_normalise)
    
    alpha_param = cp.Variable(shape = (n_task,), nonneg = True) 
    prev_alpha_param = cp.Parameter(shape = (n_task,), value = np.ones(n_task) / n_task) 
    beta_param = GTG_param @ alpha_param
    prev_beta_param = GTG_param @ prev_alpha_param
    tilde_phi_delta_alpha_param = (1/(prev_alpha_param) + GTG_param @ (1/(prev_beta_param))) @ (alpha_param - prev_alpha_param)
    phi_alpha = cp.log(alpha_param * GTG_normalise) + cp.log(beta_param)
    
    constraint = [-phi_alpha[i] <= 0 for i in range(n_task)]
    objective = cp.Minimize(tilde_phi_delta_alpha_param / GTG_normalise + cp.sum(beta_param))
    prob = cp.Problem(objective, constraint)
    for k in range(n_iter):
        try:
            prob.solve(solver=cp.ECOS, max_iters=100)
        except Exception:
            alpha_param.value = prev_alpha_param.value
        if alpha_param.value is None or alpha_param.value is np.nan:
            alpha_param.value = prev_alpha_param.value

        delta = np.linalg.norm(alpha_param.value - prev_alpha_param.value)
        gtg = GTG / GTG_normalise
        if (delta <= 1e-6 or 
            np.linalg.norm(gtg @ alpha_param.value - 1 / (alpha_param.value + 1e-10)) < 1e-3):
            break
        #print("VAL : ", alpha_param.value)
        prev_alpha_param.value = alpha_param.value.copy()

    alpha = alpha_param.value
    #alpha = alpha / sum(alpha)  # Normalize alpha to unit length
    alpha = torch.from_numpy(alpha).to(device=jacobian.device, dtype=jacobian.dtype)
    descent_direction = jacobian.t() @ alpha
    '''norm = torch.linalg.norm(descent_direction) # normalisation used by official implementation
    if norm > 1.0:
        alpha = alpha / norm
        descent_direction = jacobian.t() @ alpha'''
    return (descent_direction, alpha.numpy())



