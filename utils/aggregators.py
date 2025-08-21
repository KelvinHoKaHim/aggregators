import torch
from typing import Optional, Tuple, Union
import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers

class MGDA:
    name = "MGDA"
    
    def __call__(self,jacobian : torch.Tensor, max_iters : int = 100, epsilon: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
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
        alpha = alpha.to(device=jacobian.device)  # Ensure alpha is on the same device and dtype as jacobian
        descent_direction = jacobian.t() @ alpha
        return (descent_direction, alpha)
    

class Nash_MTL:
    name = "Nash-MTL"
    
    def __call__(self ,jacobian: torch.Tensor, prev_alpha: Optional[torch.Tensor] = None, max_norm: float = 1.0, optim_niter: int = 50, return_status : bool =  False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Computes the alpha weight vector using the 
        NashMTL algorithm based on the provided Jacobian matrix.

        This function adapts the NashMTL algorithm from a stateful class to a stateless function. It always
        computes a new alpha based on the current matrix and an optional previous alpha, applying a 
        maximum norm constraint if specified.

        Args:
            matrix (torch.Tensor): The Jacobian matrix of shape (n_tasks, n_features).
            prev_alpha (np.ndarray, optional): The previous alpha vector of shape (n_tasks,). If None,
                initialized to a vector of ones. Defaults to None.
            max_norm (float): The maximum allowed norm of matrix^T @ alpha. Defaults to 1.0.
            optim_niter (int): The number of iterations for the optimization solver. Defaults to 20.

        Returns:
            torch.Tensor: The computed alpha vector of shape (n_tasks,) on the same device and dtype as
                the input matrix.

        Example:
            >>> import torch
            >>> matrix = torch.tensor([[-4., 1., 1.], [6., 1., 1.]])
            >>> alpha = compute_nash_mtl_alpha(matrix)
            >>> print(alpha)
        """
        # Infer number of tasks from the matrix
        status = "suboptimal"
        n_tasks = jacobian.shape[0]
        # Initialize prev_alpha to ones if not provided
        if prev_alpha is None:
            prev_alpha = np.ones(n_tasks)
        else:
            prev_alpha = prev_alpha.detach().cpu().numpy()

        # Compute GTG matrix
        GTG = jacobian @ jacobian.T
        gtg = GTG.detach().cpu().numpy()
        # Define optimization problem with cvxpy parameters
        G_param = cp.Parameter(shape=(n_tasks, n_tasks))
        prvs_alpha_param = cp.Parameter(shape=(n_tasks,))
        alpha_param = cp.Variable(shape=(n_tasks,), nonneg=True)

        # Define expressions
        G_alpha = G_param @ alpha_param
        G_prvs_alpha = G_param @ prvs_alpha_param
        prvs_phi_tag = 1 / prvs_alpha_param + (1 / G_prvs_alpha) @ G_param
        phi_alpha = prvs_phi_tag @ (alpha_param - prvs_alpha_param)

        # Define objective and constraints
        objective = cp.Minimize(cp.sum(G_alpha) + phi_alpha )
        constraints = [
            -cp.log(alpha_param[i]) - cp.log(G_alpha[i]) <= 0
            for i in range(n_tasks)
        ]
        constraints.append(cp.norm(G_alpha) <= np.sqrt(n_tasks)) 
        prob = cp.Problem(objective, constraints)

        # Assign parameter values
        G_param.value = gtg

        #normalization_factor_param.value = np.array([normalization_factor])

        # Iterative optimization
        alpha_t = prev_alpha
        for _ in range(optim_niter):
            try:
                prvs_alpha_param.value = alpha_t
                alpha_param.value = alpha_t  # Warm start
                #prob.solve(solver=cp.MOSEK, warm_start=True, max_iters=100)
                prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
                #if np.linalg.matrix_rank(gtg) != n_tasks or prob.status != "optimal":
                #    #print(np.linalg.matrix_rank(gtg), prob.status)
                status = prob.status if prob.status == "optimal" else "suboptimal"
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print("CAUTION : ", prob.status)
                    alpha_t = prvs_alpha_param.value
                else:
                    alpha_t = alpha_param.value
            except Exception as e: # sometimes when lr is not small enough, an exception "TypeError: prvs_alpha_param.value must be real number" is raised. Seem to be issues internal to the solver
                print(f"Error {e}")
                alpha_t = prvs_alpha_param.value


            # Check stopping criteria
            if (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t)) < 1e-6 or 
                np.linalg.norm(alpha_t - prvs_alpha_param.value) < 1e-6):
                break

            if alpha_t is None:
                alpha_t = prev_alpha
                break

        alpha = torch.from_numpy(alpha_t).to(device=jacobian.device, dtype = torch.float32)

        # Apply clipping
        if max_norm > 0:
            norm = torch.linalg.norm(alpha @ jacobian)
            if norm > max_norm:
                alpha = (alpha / norm) * max_norm
        
        alpha = alpha.to(device=jacobian.device)
        direction = alpha @ jacobian

        if return_status:
            return (direction, alpha, status)
        else:
            return (direction, alpha)
        

class Nash_MTL_star:
    name = "Nash-MTL*"
    def __call__(self,jacobian: torch.Tensor, prev_alpha: Optional[torch.Tensor] = None, max_norm: float = 1.0, optim_niter: int = 20, return_status : bool =  False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Computes the alpha weight vector using the 
        NashMTL algorithm based on the provided Jacobian matrix.

        This function adapts the NashMTL algorithm from a stateful class to a stateless function. It always
        computes a new alpha based on the current matrix and an optional previous alpha, applying a 
        maximum norm constraint if specified.

        Args:
            matrix (torch.Tensor): The Jacobian matrix of shape (n_tasks, n_features).
            prev_alpha (np.ndarray, optional): The previous alpha vector of shape (n_tasks,). If None,
                initialized to a vector of ones. Defaults to None.
            max_norm (float): The maximum allowed norm of matrix^T @ alpha. Defaults to 1.0.
            optim_niter (int): The number of iterations for the optimization solver. Defaults to 20.

        Returns:
            torch.Tensor: The computed alpha vector of shape (n_tasks,) on the same device and dtype as
                the input matrix.

        Example:
            >>> import torch
            >>> matrix = torch.tensor([[-4., 1., 1.], [6., 1., 1.]])
            >>> alpha = compute_nash_mtl_alpha(matrix)
            >>> print(alpha)
        """
        # Infer number of tasks from the matrix
        status = "suboptimal"
        n_tasks = jacobian.shape[0]
        
        # Initialize prev_alpha to ones if not provided
        if prev_alpha is None:
            prev_alpha = np.ones(n_tasks)
        else:
            prev_alpha = prev_alpha.detach().cpu().numpy()


        # Compute GTG matrix
        GTG = jacobian @ jacobian.T
        gtg = GTG.detach().cpu().numpy() 

        # Define optimization problem with cvxpy parameters
        G_param = cp.Parameter(shape=(n_tasks, n_tasks))
        prvs_alpha_param = cp.Parameter(shape=(n_tasks,))
        alpha_param = cp.Variable(shape=(n_tasks,), nonneg=True)

        # Define expressions
        G_alpha = G_param @ alpha_param
        G_prvs_alpha = G_param @ prvs_alpha_param
        prvs_phi_tag = 1 / prvs_alpha_param + (1 / G_prvs_alpha) @ G_param
        phi_alpha = prvs_phi_tag @ (alpha_param - prvs_alpha_param)

        # Define objective and constraints
        objective = cp.Minimize(cp.sum(G_alpha) + phi_alpha )
        constraints = [
            -cp.log(alpha_param[i]) - cp.log(G_alpha[i]) <= 0
            for i in range(n_tasks)
        ]
        constraints.append(cp.norm(G_alpha) <= np.sqrt(n_tasks)) 
        prob = cp.Problem(objective, constraints)

        # Assign parameter values
        G_param.value = gtg
        #normalization_factor_param.value = np.array([normalization_factor])

        # Iterative optimization
        alpha_t = prev_alpha
        for _ in range(optim_niter):
            try:
                prvs_alpha_param.value = alpha_t
                alpha_param.value = alpha_t  # Warm start
                #prob.solve(solver=cp.MOSEK, warm_start=True, max_iters=100)
                prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
                #if np.linalg.matrix_rank(gtg) != n_tasks or prob.status != "optimal":
                #    #print(np.linalg.matrix_rank(gtg), prob.status)
                status = prob.status if prob.status == "optimal" else "suboptimal"
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print("CAUTION : ", prob.status)
                    alpha_t = prvs_alpha_param.value
                else:
                    alpha_t = alpha_param.value
            except Exception as e: # sometimes when lr is not small enough, an exception "TypeError: prvs_alpha_param.value must be real number" is raised. Seem to be issues internal to the solver
                print(f"Error {e}")
                alpha_t = prvs_alpha_param.value

            # Check stopping criteria
            if (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t)) < 1e-6 or 
                np.linalg.norm(alpha_t - prvs_alpha_param.value) < 1e-6):
                break

            if alpha_t is None:
                alpha_t = prev_alpha
                break

        # Convert to tensor
        alpha = torch.from_numpy(alpha_t).to(device=jacobian.device, dtype=jacobian.dtype)

        alpha = alpha / torch.sum(alpha) # normalisation instead of max norm
        alpha = alpha.to(device=jacobian.device)

        direction = alpha @ jacobian
        if return_status:
            return (direction, alpha, status)
        else:
            return (direction, alpha)        

class UPGrad:
    name = "UPGrad"
    
    def __call__(self,jacobian : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
        # Output : descent direction of length d, and alpha of length n.
        m = len(jacobian)
        jac = jacobian.detach().cpu().numpy()
        G = jac @ jac.T

        #G_norm = np.trace(G)
        #G = G / G_norm  # Normalize the Gramian matrix, avoid numerical issues when entries of G is too small
        G = G.astype(np.double)


        # UPgrad projects each gradient to dual cone of the Jacobian, to get projected gradients
        # You average the projected gradients (implementation-wise, you may be averaging the alphas)
        # That projection will be in the confe as well (alphas are non-negative)

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
            sol = solvers.qp(P, q, G_cvx, h, A, b, options={'show_progress': False})  # UPGrad is linear under scaling
            w = np.array(sol['x']).squeeze()
            alpha = alpha + w

        alpha = torch.from_numpy(alpha).to(device=jacobian.device, dtype=jacobian.dtype)
        direction = jacobian.t() @ alpha
        return (direction, alpha)

class UPGrad_star:
    name = "UPGrad*"
    
    def __call__(self,jacobian : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
        # Output : descent direction of length d, and alpha of length n.
        m = len(jacobian)
        #G = jacobian @ jacobian.T
        jac = jacobian.detach().cpu().numpy()
        G = jac @ jac.T

        #G_norm = np.trace(G)
        #G = G / G_norm  # Normalize the Gramian matrix, avoid numerical issues when entries of G is too small
        G = G.astype(np.double)

        # UPgrad projects each gradient to dual cone of the Jacobian, to get projected gradients
        # You average the projected gradients (implementation-wise, you may be averaging the alphas)
        # That projection will be in the confe as well (alphas are non-negative)

        P = matrix(G)
        q = matrix(np.zeros(m))
        A = None
        b = None
        G_cvx = matrix(-1.0*np.eye(m))
        alpha = np.zeros(m)
        for i in range(m): # project each gradient to the dual cone, then add up the projected gradients
            neg_e_i = np.zeros(m)  
            neg_e_i[i] = -1.0 / m                   # Minimize v^T JJ^T v 
            h = matrix(neg_e_i)                     # Subject to v \preceq e_i / n
            sol = solvers.qp(P, q, G_cvx, h, A, b, options={'show_progress': False})  # UPGrad is linear under scaling
            w = np.array(sol['x']).squeeze()
            alpha = alpha + w

        alpha = torch.from_numpy(alpha).to(device=jacobian.device, dtype=jacobian.dtype)
        alpha = alpha / torch.sum(alpha)  # Normalize alpha 
        direction = jacobian.t() @ alpha
        return (direction, alpha)

class DualProj:
    name = "DualProj"
    
    def __call__(self,jacobian : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
        # Output : descent direction of length d, and alpha of length n.
        jac = jacobian.detach().cpu().numpy()
        m = len(jac)
        G = jac @ jac.T 

        #G_norm = np.sum(np.diag(G)) #normalization to ensure positive definiteness and full rank
        #G = G / G_norm
        G = G.astype(np.double)

        P = matrix(G)                           # Minimize v^T JJ^T v
        q = matrix(np.zeros(m))                 # with constraint
        G_cvx = matrix(-1.0 * np.eye(m))        # v_i >= 1 for each i
        h = matrix((-1.0 / m) * np.ones(m))     
        A = None 
        b = None
        sol = solvers.qp(P, q, G_cvx, h, A, b, options={'show_progress': False})
        alpha = np.array(sol['x']).squeeze()

        alpha = torch.from_numpy(alpha).to(device=jacobian.device, dtype=jacobian.dtype)
        direction = jacobian.t() @ alpha
        return (direction, alpha)
    

class DualProj_star:
    name = "DualProj*"
    
    def __call__(self,jacobian : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input : a  nxd jacobian, where n is the number of task, d is the dimension of x (set to 4 by default)
        # Output : descent direction of length d, and alpha of length n.
        jac = jacobian.detach().cpu().numpy()
        m = len(jac)
        G = jac @ jac.T 

        #G_norm = np.sum(np.diag(G)) #normalization to ensure positive definiteness and full rank
        #G = G / G_norm
        G = G.astype(np.double)


        P = matrix(G)                           # Minimize v^T JJ^T v
        q = matrix(np.zeros(m))                 # with constraint
        G_cvx = matrix(-1.0 * np.eye(m))        # v_i >= 1 for each i
        h = matrix((-1.0 / m) * np.ones(m))     
        A = None 
        b = None
        sol = solvers.qp(P, q, G_cvx, h, A, b, options={'show_progress': False})
        alpha = np.array(sol['x']).squeeze()


        alpha = torch.from_numpy(alpha).to(device=jacobian.device, dtype=jacobian.dtype)
        alpha = alpha / torch.sum(alpha)
        direction = jacobian.t() @ alpha
        return (direction, alpha)

aggregator_dict = {
        "MGDA" : MGDA,
        "Nash-MTL" : Nash_MTL,
        "Nash-MTL*" : Nash_MTL_star,
        "UPGrad" : UPGrad,
        "UPGrad*" : UPGrad_star,
        "DualProj" : DualProj,
        "DualProj*" : DualProj_star
    }