import os
import torch
import numpy as np
from typing import List, Tuple
import argparse
from utils.aggregators import MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star, aggregator_dict
from utils.synthetic_problems import vlmop2, omnitest, problem_dict
import pandas as pd

def find_jacobian(x: torch.Tensor, func: List[torch.Tensor]) -> torch.Tensor:
    """Compute the Jacobian matrix for the given function and input."""
    jac = []
    for func_val in func:
        if x.grad is not None:
            x.grad.zero_()
        func_val.backward(retain_graph=True)
        jac.append(x.grad.clone().detach())
    return torch.stack(jac)

def iterate(problem, aggregator, seed, epochs, lr, eps) -> Tuple[np.array]:
    torch.manual_seed(seed)
    curr_problem = problem()  
    curr_aggregator = aggregator()  
    mgda = MGDA() # used for computing measure to Pareto stationarity
    # Initialize starting point within bounds

    #To avoid reaching the boundary, we restrict the range of possible initial values to half of the original optimization region.
    mu = (curr_problem.lower_bound + curr_problem.upper_bound) / 2  # the mid point of the desired range 
    epsilon = (curr_problem.upper_bound - curr_problem.lower_bound) / 4 # the radius of the desired range
    x = mu - epsilon + 2 * epsilon * torch.rand(curr_problem.dimension)  
    prev_alpha = torch.ones(curr_problem.n_task)
    x.requires_grad = True

    # Initialize tracking lists
    x_trajectory = []
    y_trajectory = []
    track_d, track_measure_to_PS, track_alpha = [], [], []
    try:
        for iter_count in range(epochs):
            y = curr_problem(x)
            jacobian = find_jacobian(x, y)
            if aggregator.name == "Nash-MTL" or aggregator.name == "Nash-MTL*":   # Nash-MTL is unstable if it does not optimize from prev_alpha
                d, alpha = curr_aggregator(jacobian, prev_alpha)
                prev_alpha = alpha
            else:
                d, alpha = curr_aggregator(jacobian)

            d_mgda, alpha_mgda = mgda(jacobian) # used for computing measure to Pareto stationary 

            with torch.no_grad(): 
                x -= lr * d

                # clip x back to the domain if needed
                if torch.any((x < curr_problem.lower_bound) | (x > curr_problem.upper_bound)):
                    x.clamp_(min=curr_problem.lower_bound, max=curr_problem.upper_bound)

            # logging all data
            x_trajectory.append(x.detach().cpu().numpy())
            y_trajectory.append(np.array([yi.item() for yi in y]))
            norm_d = torch.norm(d).item()
            norm_d_mgda = torch.norm(d_mgda).item()
            norm_alpha = torch.norm(alpha).item()
            track_d.append(norm_d)
            track_alpha.append(norm_alpha)
            track_measure_to_PS.append(norm_d_mgda)

            # progress logging 
            if iter_count+1 % 1000 == 0:
                print(f"{aggregator.name} on {problem.name} with seed {seed}.")
                print(f"Epoch = {iter_count+1}/{epochs}, norm_d = {norm_d}, norm_alpha = {norm_alpha}, norm_d_mgda = {norm_d_mgda}")
            
            # in case of hitting boundaries, the algorithm could be stuck at the same place despite not meeting the stopping criteria
            # we drop the last condition for now since our seeds are chosen to not hit any boundaries
            #if norm_d < eps or norm_d_mgda < eps or torch.linalg.vector_norm(x - prev_x) < 1e-7: 
            if norm_d < eps or norm_d_mgda < eps: 
                break
        loss_log = ". ".join([f"Loss{i+1}:{y[i].item()}" for i in range(len(y))])
        log = f"{aggregator.name}: Training ended at eppoch {iter_count+1}/{epochs}. {loss_log}, d_MGDA = {norm_d_mgda}"
    except Exception as e:
        e = str(e)
        x_trajectory = [-1.0*np.ones(curr_problem.dimension) for i in range(epochs)]
        y_trajectory = [-1.0*np.ones(curr_problem.n_task) for i in range(epochs)]
        track_d = [-1 for i in range(epochs) for i in range(epochs)]
        track_alpha = [-1 for i in range(epochs) for i in range(epochs)]
        track_measure_to_PS = [-1 for i in range(epochs) for i in range(epochs)]
        loss_log = ". ".join([f"Loss{i+1}:{y[i].item()}" for i in range(len(y))])
        log = f"{aggregator.name}: Training interrupted at {iter_count+1}/{epochs} with message '{e}'. {loss_log}, d_MGDA = {norm_d_mgda}"

        
    x_trajectory = list(x_trajectory) # stored as list so that they can be stored in a pandas dataframe
    y_trajectory = list(y_trajectory)
    track_d = np.array(track_d)
    track_alpha = np.array(track_alpha)
    track_measure_to_PS = np.array(track_measure_to_PS)
    return (x_trajectory, y_trajectory, track_d, track_alpha, track_measure_to_PS, log)

def run_experiment(problems : List, aggregators : List, seeds : List[int], lr : float, epochs : int, eps : float):
    results_path = "synthetic_benchmark/results"

    # print experiment details
    print(20*"=" + "Experiment Details" + 20*"=")
    print(f"Aggregators :{[aggregator.name for aggregator in aggregators]}")
    print(f"Synthetic problems :{[problem.name for problem in problems]}")
    print(f"Seeds :{seeds}")
    print(f"Learing rate :{lr}")
    print(f"Number of epochs :{epochs}")
    print(f"Tolorance :{eps}")
    print(58*"=")

    os.makedirs(results_path, exist_ok=True)  

    # record experiment parameters in a txt file
    with open(f"{results_path}/experiment_details.txt", "w") as f: 
        print(20*"=" + "Experiment Details" + 20*"=", file = f)
        print(f"Aggregators: {[aggregator.name for aggregator in aggregators]}", file = f)
        print(f"Synthetic problems: {[problem.name for problem in problems]}", file = f)
        print(f"Seeds: {seeds}", file = f)
        print(f"Learing rate :{lr}", file = f)
        print(f"Number of epoch {epochs}", file = f)
        print(f"Tolorance :{eps}", file = f)
        print(58*"=", file = f)
    
    
    for problem in problems:
        problem_folder_path = os.path.join(results_path, problem.name)
        os.makedirs(problem_folder_path, exist_ok=True)
        for seed in seeds:
            seed_folder_path = os.path.join(problem_folder_path, f"seed_{seed}")
            os.makedirs(seed_folder_path, exist_ok=True)
            data_folder_path = os.path.join(seed_folder_path, "data")
            os.makedirs(data_folder_path, exist_ok=True)

            # Collect results for all aggregators
            for aggregator in aggregators:
                print(40*"=")
                print("Now testing with...")
                print(f"Aggregator : {aggregator.name}")
                print(f"Synthetic problem : {problem.name}")
                print(f"Seed : {seed}")
                x_trajectory, y_trajectory, track_norm_d, track_alpha, track_measure_to_PS, log = iterate(problem, aggregator, seed, epochs, lr, eps)
 
                with open(os.path.join(seed_folder_path, "log.txt"), "a") as f:
                    f.write(log + "\n")

                n_iteration = np.arange(len(track_norm_d))
                df = pd.DataFrame({
                    "n_iteration": n_iteration,
                    "x_trajectory": x_trajectory,
                    "y_trajectory": y_trajectory,
                    "norm_d": track_norm_d,
                    "alpha": track_alpha,
                    "measure_PS": track_measure_to_PS,
                })
                csv_filename = os.path.join(data_folder_path, f"{aggregator.name}.csv")
                pickle_filename = os.path.join(data_folder_path, f"{aggregator.name}.pkl")
                df.to_csv(csv_filename)
                df.to_pickle(pickle_filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for running synthetic problem benchmark")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=400000, help="Number of epochs")
    parser.add_argument("--eps", type=float, default=1e-2, help="Stopping criteria tolorance")
    parser.add_argument(
        "--aggregators", 
        type=str, 
        nargs="+",
        default=["MGDA", "UPGrad", "UPGrad*", "DualProj", "DualProj*","Nash-MTL", "Nash-MTL*"],  # Default list of aggregators
        help="List of aggregators to be used in experiment"
    )
    parser.add_argument(
        "--problems", 
        type=str, 
        nargs="+",
        default=["VLMOP2", "Omnitest"],  # Default list of aggregators
        help="List of synthetic problems to be used in experiment"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+",
        default=[24, 42, 48, 100, 123],  # Default random seeds
        help="List of randoms seeds"
    )

    args = parser.parse_args()

    aggregators = [aggregator_dict[aggregators_name] for aggregators_name in args.aggregators if aggregators_name in aggregator_dict]
    problems = [problem_dict[problem_name] for problem_name in args.problems if problem_name in problem_dict]
    seeds = args.seeds
    lr = args.lr 
    epochs = args.epochs
    eps = args.eps
    run_experiment(problems, aggregators, seeds, lr, epochs, eps)
