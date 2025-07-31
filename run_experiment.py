import torch
import numpy as np
from mgda import MGDA
from nash_mtl import NASH_MTL
from upgrad import UPGrad
from dualproj import DualProj
import pickle
import time
from typing import List, Callable, Union, Tuple
from matplotlib import pyplot as plt
from torchjd.aggregation import MGDA as TorchJD_MGDA
from torchjd.aggregation import NashMTL as TorchJD_NashMTL
from torchjd.aggregation import UPGrad as TorchJD_UPGrad
from torchjd.aggregation import DualProj as TorchJD_DualProj
import matplotlib
import os
import pandas as pd

# Define bounds for each optimization problem
bounds = {
    "vlmop2": (-2.0, 2.0),
    "zdt2": (0.0, 1.0),
    "zdt3": (0.0, 1.0),
    "zdt4": (0.0, 1.0),
    "zdt1": (0.0, 1.0),
    "omnitest": (0.0, 6.0)
}

# Optimization problem definitions
def vlmop2(x):  # Dimension n=4
    n = len(x)
    f1 = 1 - torch.exp(-1.0 * torch.sum((x - 1 / np.sqrt(n)) ** 2))
    f2 = 1 - torch.exp(-1.0 * torch.sum((x + 1 / np.sqrt(n)) ** 2))
    return ([f1, f2], -2.0, 2.0)

def zdt4(x):
    f1 = x[0]
    g = 1 + 10 * (len(x) - 1) + torch.sum(torch.pow(x[1:], 2) - 10 * torch.cos(4 * np.pi * x[1:]))
    f2 = g * (1 - torch.sqrt(f1 / g))
    return ([f1, f2], 0.0, 1.0)

def zdt3(x):  # Dimension n=30
    f1 = x[0]
    g = 1 + 9 * torch.mean(x[1:])
    f2 = g * (1 - torch.sqrt(f1 / g) - f1 * torch.sin(10 * np.pi * f1) / g)
    return ([f1, f2], 0.0, 1.0)

def zdt2(x):  # Dimension n=30
    f1 = x[0]
    g = 1 + 9 * torch.mean(x[1:])
    f2 = g * (1 - torch.pow(f1 / g, 2))
    return ([f1, f2], 0.0, 1.0)

def omnitest(x):  # Dimension n=3, bounds 0 to 6
    f1 = torch.sum(torch.sin(np.pi * x))
    f2 = torch.sum(torch.cos(np.pi * x))
    return ([f1, f2], 0.0, 6.0)

def find_jacobian(x: torch.Tensor, func: List[torch.Tensor]):
    """Compute the Jacobian matrix for the given function and input."""
    jac = []
    for func_val in func:
        if x.grad is not None:
            x.grad.zero_()
        func_val.backward(retain_graph=True)
        jac.append(x.grad.clone().detach())
    return torch.stack(jac)

def iterate(problem, aggregator, torchjd_aggregator, seed, n_task, max_iter, eta, dimension, normalize_d: bool = False, eps=1e-2):
    """Run the optimization loop for a problem using the specified aggregator."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    lb, ub = bounds[problem.__name__]
    
    # Initialize starting point within bounds
    mu = (lb + ub) / 2
    epsilon = (ub - lb) / 4
    x = mu - epsilon + 2 * epsilon * torch.rand(dimension)
    x.requires_grad = True

    # Initialize tracking arrays
    x_trajectory = np.zeros(shape=(0, dimension))
    y_trajectory = np.zeros(shape=(0, n_task))
    track_alpha, track_norm_d, track_distance_PS, track_difference, track_cos = [], [], [], [], []
    start_hitting_iterations, stop_hitting_iterations = [], []
    iter_count, unchanged_count = 0, 0
    was_clamped = False
    x_old = None

    while True:
        # Evaluate problem and store trajectories
        y, lower, upper = problem(x)
        x_trajectory = np.vstack((x_trajectory, x.detach().cpu().numpy()))
        y_tensor = torch.stack([yi.view(1) for yi in y]).squeeze()
        y_trajectory = np.vstack((y_trajectory, y_tensor.detach().cpu().numpy()))

        # Compute descent direction
        jacobian = find_jacobian(x, y)
        d, alpha = aggregator(jacobian)

        # Compare with torchjd aggregator if provided
        if torchjd_aggregator:
            try:
                torchjd_aggregator.reset()
            except:
                pass
            torchjd_d = torchjd_aggregator(jacobian)
            track_difference.append(torch.norm(d - torchjd_d).item())
            if aggregator.__name__ == "NASH_MTL":
                d = torchjd_d  # Use official implementation for NASH_MTL for now
            cos = (d.type(torch.float32) @ torchjd_d / (torch.norm(d) * torch.norm(torchjd_d))).item()
            track_cos.append(cos)
            if cos < 0.9:
                print(f"Warning: Low cosine similarity {cos} at iteration {iter_count}")
        else:
            track_difference.append(0.0)

        # Normalize descent direction if specified
        if normalize_d and np.sum(alpha) > 0:
            d = d / np.sum(alpha)
        track_alpha.append(alpha)
        norm_d = torch.norm(d).item()
        track_norm_d.append(norm_d)
        track_distance_PS.append(torch.norm(MGDA(jacobian)[0]).item())

        # Update x and clamp within bounds
        with torch.no_grad():
            x -= eta * d
            prev = x.clone()
            if lower is not None and upper is not None:
                x[x < lower] = lower
                x[x > upper] = upper
                is_clamped = not torch.equal(x, prev)
                if is_clamped and not was_clamped:
                    start_hitting_iterations.append(iter_count)
                elif not is_clamped and was_clamped:
                    stop_hitting_iterations.append(iter_count)
                was_clamped = is_clamped

            # Check convergence. It is possible that x remain unchanged despite d is non-zero if x touches the boundary
            if x_old is not None and torch.allclose(x, x_old, rtol=1e-5, atol=1e-8):
                unchanged_count += 1
            else:
                unchanged_count = 0
            x_old = x.clone()

        # Stopping conditions
        if unchanged_count >= 20 or track_distance_PS[-1] <= eps or iter_count >= max_iter: 
            break

        # Progress logging
        if iter_count % 1000 == 0:
            print(f"normalize_d={normalize_d}, seed={seed}, {aggregator.__name__}-{problem.__name__}, iter={iter_count}, norm_d={norm_d:.2e}, distance_PS={track_distance_PS[-1]:.2e}")
        iter_count += 1

    # Convert lists to arrays for return
    return (x_trajectory, y_trajectory, 
            np.array(track_norm_d), np.array(track_alpha), np.array(track_distance_PS), 
            np.array(track_difference), np.array(track_cos), start_hitting_iterations, stop_hitting_iterations)

def run_experiment(problem, aggregator, dimension, max_iter, eta, seed, n_task, normalize_d: bool = False):
    """Set up and run an optimization experiment."""
    torchjd_map = {
        "MGDA": TorchJD_MGDA(),
        "NASH_MTL": TorchJD_NashMTL(n_tasks=n_task),
        "UPGrad": TorchJD_UPGrad(),
        "DualProj": TorchJD_DualProj()
    }
    torchjd_aggregator = torchjd_map.get(aggregator.__name__, None)
    if not torchjd_aggregator and aggregator.__name__ not in torchjd_map:
        raise ValueError("Unknown aggregator")
    return iterate(problem, aggregator, torchjd_aggregator, seed, n_task, max_iter, eta, dimension, normalize_d)

def save_data(seed_folder, aggregator_name, normalize_d, x_trajectory, y_trajectory, track_norm, track_alpha, track_distance_PS, track_difference, track_cos, start_hitting, stop_hitting):
    """Save experiment results to CSV and pickle files."""
    norm_str = 'normalized' if normalize_d else 'unnormalized'
    n_iterations = len(track_norm)
    iterations = np.arange(n_iterations)
    dimension, n_task = x_trajectory.shape[1], y_trajectory.shape[1]

    # Create DataFrame for CSV
    df = pd.DataFrame({'iteration': iterations})
    for i in range(dimension):
        df[f'x_{i+1}'] = x_trajectory[:, i]
    for i in range(n_task):
        df[f'y_{i+1}'] = y_trajectory[:, i]
    df['norm_d'] = track_norm
    df['distance_PS'] = track_distance_PS
    df['difference'] = track_difference
    df['cos'] = track_cos
    for i in range(n_task):
        df[f'alpha_{i+1}'] = track_alpha[:, i]
    df['is_start_hitting'] = False
    df['is_stop_hitting'] = False
    for idx in start_hitting:
        df.loc[idx, 'is_start_hitting'] = True
    for idx in stop_hitting:
        df.loc[idx, 'is_stop_hitting'] = True
    df.to_csv(os.path.join(seed_folder, f"{aggregator_name}_{norm_str}.csv"), index=False)

    # Save to pickle
    pickle_data = {
        'x_trajectory': x_trajectory, 'y_trajectory': y_trajectory, 'track_norm_d': track_norm,
        'track_alpha': track_alpha, 'track_distance_PS': track_distance_PS, 'track_difference': track_difference,
        'track_cos': track_cos, 'start_hitting_iterations': start_hitting, 'stop_hitting_iterations': stop_hitting,
        'final_x': x_trajectory[-1], 'final_y': y_trajectory[-1]
    }
    with open(os.path.join(seed_folder, f"{aggregator_name}_{norm_str}.pkl"), 'wb') as f:
        pickle.dump(pickle_data, f)

if __name__ == "__main__":
    # Configuration
    problem_dim = {"vlmop2": 4, "zdt3": 30, "zdt2": 30, "omnitest": 3}
    max_iter, eta, n_task = 20000, 0.001, 2
    seeds = [24]
    aggregator_colors = {"MGDA": 'blue', "UPGrad": 'green', "DualProj": 'red', "NASH_MTL": 'orange'}
    aggregator_colors_normalized = {"MGDA": 'blue', "UPGrad": 'lime', "DualProj": 'magenta', "NASH_MTL": 'gold'}

    # Setup result directory
    os.makedirs("experiment results", exist_ok=True)

    for problem in [zdt2]:  # Add more problems here as needed
        problem_folder = os.path.join("experiment results", problem.__name__)
        os.makedirs(problem_folder, exist_ok=True)
        dimension = problem_dim[problem.__name__]

        for seed in seeds:
            seed_folder = os.path.join(problem_folder, f"seed_{seed}")
            os.makedirs(seed_folder, exist_ok=True)

            # Initialize plotting figures
            fig_unnorm, axes_unnorm = plt.subplots(2, 2, figsize=(12, 12))  # Unnormalized plots
            fig_combined2, axes_combined2 = plt.subplots(2, 4, figsize=(24, 12))  # Normalized vs unnormalized
            results = {agg.__name__: {False: None, True: None} for agg in [MGDA, UPGrad, DualProj, NASH_MTL]}

            for aggregator in [MGDA, UPGrad, DualProj, NASH_MTL]: # run experiment
                for normalize_d in [False, True]:
                    print(f"Running {aggregator.__name__} on {problem.__name__}, seed={seed}, normalize_d={normalize_d}")
                    result = run_experiment(problem, aggregator, dimension, max_iter, eta, seed, n_task, normalize_d)
                    x_trajectory, y_trajectory, track_norm, track_alpha, track_distance_PS, track_difference, track_cos, start_hitting, stop_hitting = result
                    
                    # Save experiment data
                    save_data(seed_folder, aggregator.__name__, normalize_d, *result)

                    # Prepare plotting data
                    f1, f2 = y_trajectory.T
                    n_iteration = np.arange(len(f1))
                    results[aggregator.__name__][normalize_d] = {
                        'n_iteration': n_iteration, 'f1': f1, 'f2': f2, 'track_norm': track_norm,
                        'track_distance_PS': track_distance_PS, 'start_hitting': start_hitting, 'stop_hitting': stop_hitting
                    }
                    color = aggregator_colors_normalized[aggregator.__name__] if normalize_d else aggregator_colors[aggregator.__name__]
                    label = f"{aggregator.__name__}{'*' if (normalize_d and aggregator.__name__ != 'MGDA') else ''}"

                    # Plot unnormalized results (excluding MGDA normalized)
                    if aggregator.__name__ != "MGDA" or not normalize_d:
                        for ax, data_key in zip(axes_unnorm.flat, ['f1', 'f2', 'track_norm', 'track_distance_PS']):
                            ax.plot(n_iteration, results[aggregator.__name__][normalize_d][data_key], color=color, label=label)
                            if start_hitting:
                                ax.plot(start_hitting, results[aggregator.__name__][normalize_d][data_key][start_hitting], 'ro')
                            if stop_hitting:
                                ax.plot(stop_hitting, results[aggregator.__name__][normalize_d][data_key][stop_hitting], 'go')

                    # Plot combined normalized and unnormalized results
                    row = 1 if normalize_d else 0
                    for ax, data_key in zip(axes_combined2[row], ['f1', 'f2', 'track_norm', 'track_distance_PS']):
                        ax.plot(n_iteration, results[aggregator.__name__][normalize_d][data_key], color=color, label=label)
                        if start_hitting:
                            ax.plot(start_hitting, results[aggregator.__name__][normalize_d][data_key][start_hitting], 'ro')
                        if stop_hitting:
                            ax.plot(stop_hitting, results[aggregator.__name__][normalize_d][data_key][stop_hitting], 'go')

            # Configure and save plots
            titles = ["First Loss Function", "Second Loss Function", "Norm of d", "Measure of Pareto stationarity"]
            for ax, title in zip(axes_unnorm.flat, titles):
                ax.set_title(title)
                ax.legend()
            for col, title in enumerate(titles):
                axes_combined2[0, col].set_title(f"{title} (Unnormalized)")
                axes_combined2[1, col].set_title(f"{title} (Normalized)")
                axes_combined2[0, col].legend()
                axes_combined2[1, col].legend()

            fig_unnorm.suptitle(f"Results for {problem.__name__}, seed {seed}, eta={eta}")
            fig_combined2.suptitle(f"Results for {problem.__name__}, seed {seed}, eta={eta}")
            fig_unnorm.savefig(os.path.join(seed_folder, "combined.png"), dpi=300, bbox_inches='tight')
            fig_combined2.savefig(os.path.join(seed_folder, "combined2.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_unnorm)
            plt.close(fig_combined2)

            # Save individual unnormalized plots
            for plot_type, data_key in [("First_Loss_Function_Combined", 'f1'), ("Second_Loss_Function_Combined", 'f2'),
                                        ("Norm_of_d_Combined", 'track_norm'), ("Measure_of_Pareto_Stationarity_Combined", 'track_distance_PS')]:
                fig, ax = plt.subplots(figsize=(6, 6))
                for agg_name in results:
                    if agg_name == "MGDA" and True in results[agg_name]:
                        continue
                    data = results[agg_name][False]
                    color = aggregator_colors[agg_name]
                    label = agg_name
                    ax.plot(data['n_iteration'], data[data_key], color=color, label=label)
                    if data['start_hitting']:
                        ax.plot(data['start_hitting'], data[data_key][data['start_hitting']], 'ro')
                    if data['stop_hitting']:
                        ax.plot(data['stop_hitting'], data[data_key][data['stop_hitting']], 'go')
                ax.set_xlabel("Iterations")
                ax.set_ylabel(plot_type.replace('_', ' '))
                ax.set_title(f"{plot_type.replace('_', ' ')} for {problem.__name__}, seed {seed}")
                ax.legend()
                fig.savefig(os.path.join(seed_folder, f"{plot_type}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

            # Save individual normalized and unnormalized plots
            for norm_str, norm_flag in [("Unnormalized", False), ("Normalized", True)]:
                for plot_type, data_key in [
                    ("First_Loss_Function", 'f1'),
                    ("Second_Loss_Function", 'f2'),
                    ("Norm_of_d", 'track_norm'),
                    ("Measure_of_Pareto_Stationarity", 'track_distance_PS')
                ]:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    for agg_name in results:
                        # Skip MGDA for normalized plots if it's not intended to be plotted separately
                        if agg_name == "MGDA" and norm_flag:
                            continue
                        data = results[agg_name][norm_flag]
                        color = aggregator_colors_normalized[agg_name] if norm_flag else aggregator_colors[agg_name]
                        label = f"{agg_name}{'*' if norm_flag else ''}"
                        ax.plot(data['n_iteration'], data[data_key], color=color, label=label)
                        if data['start_hitting']:
                            ax.plot(data['start_hitting'], data[data_key][data['start_hitting']], 'ro')
                        if data['stop_hitting']:
                            ax.plot(data['stop_hitting'], data[data_key][data['stop_hitting']], 'go')
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel(plot_type.replace('_', ' '))
                    ax.set_title(f"{plot_type.replace('_', ' ')} ({norm_str}) for {problem.__name__}, seed {seed}")
                    ax.legend()
                    fig.savefig(os.path.join(seed_folder, f"{plot_type}_{norm_str}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)