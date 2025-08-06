import torch
import numpy as np
from typing import List, Callable, Union, Tuple
from matplotlib import pyplot as plt
from aggregators import MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star
import os
import pandas as pd
from synthetic_problems import zdt3, vlmop2, omnitest

problems = [vlmop2, omnitest, zdt3]
aggregators = [MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star]
#aggregators = [Nash_MTL]
seeds = [24, 42, 48, 100, 123]
seeds.sort() 

def find_jacobian(x: torch.Tensor, func: List[torch.Tensor]):
    """Compute the Jacobian matrix for the given function and input."""
    jac = []
    for func_val in func:
        if x.grad is not None:
            x.grad.zero_()
        func_val.backward(retain_graph=True)
        jac.append(x.grad.clone().detach())
    return torch.stack(jac)

def iterate(problem, aggregator, seed, max_iter=15000, eta=0.001, eps=1e-2):
    torch.manual_seed(seed)
    curr_problem = problem()  # initialise the synthetic problem 
    curr_aggregator = aggregator()  # initialise the aggregator
    mgda = MGDA()
    decay = False
    # Initialize starting point within bounds
    mu = (curr_problem.lower_bound + curr_problem.upper_bound) / 2  
    epsilon = (curr_problem.upper_bound - curr_problem.lower_bound) / 4
    x = mu - epsilon + 2 * epsilon * torch.rand(curr_problem.dimension)  # restrict field of possible values x can take to 1/2 of the original range
    prev_alpha = torch.ones(curr_problem.dimension)
    x.requires_grad = True

    # Initialize tracking lists
    x_trajectory = []
    y_trajectory = []
    track_norm_d, track_distance_PS, track_alpha = [], [], []
    start_hitting_iterations, stop_hitting_iterations = [], []
    need_clipping_prev = False

    for iter_count in range(max_iter):
        y = curr_problem(x)
        jacobian = find_jacobian(x, y)
        try:
            d, alpha = curr_aggregator(jacobian, prev_alpha = prev_alpha)
        except:
            d, alpha = curr_aggregator(jacobian)
        d_mgda, alpha_mgda = mgda(jacobian)
        norm_d_mgda = torch.norm(d_mgda).item()
        if aggregator.scheduling and norm_d_mgda < 0.05:  # handle scheduling
            if not decay:
                base_value = iter_count
                decay = True
            #lr = eta * np.exp(-0.001 * (iter_count - base_value))
            lr = eta * (0.99 * np.cos(np.pi * (iter_count - base_value) / 500)**2 + 0.01)
            print("curr lr = ", lr)
        else:
            lr = eta 

        with torch.no_grad(): 
            x -= lr * d
            # clip x back to the domain if needed
            need_clipping = torch.any((x < curr_problem.lower_bound) | (x > curr_problem.upper_bound))
            if need_clipping:
                #print("CLIPPING")
                x.clamp_(min=curr_problem.lower_bound, max=curr_problem.upper_bound)

        # logging all data
        x_trajectory.append(x.detach().cpu().numpy())
        y_trajectory.append([yi.item() for yi in y])
        norm_d = torch.norm(d).item()
        norm_alpha = torch.norm(alpha).item()
        track_norm_d.append(norm_d)
        track_alpha.append(norm_alpha)
        track_distance_PS.append(norm_d_mgda)
        if not need_clipping_prev and need_clipping:
            start_hitting_iterations.append(iter_count)
        
        if need_clipping_prev and not need_clipping:
            stop_hitting_iterations.append(iter_count)
        need_clipping_prev = need_clipping

        # progress logging 
        if iter_count % 1000 == 0:
            print(f"{aggregator.name} on {problem.name} with seed {seed}.")
            print(f"Epoch = {iter_count}, norm_d = {norm_d}, norm_alpha = {norm_alpha}, norm_d_mgda = {norm_d_mgda}, clipping = {need_clipping}")
        
        if norm_d < eps or norm_d_mgda < eps: #or torch.linalg.vector_norm(x - prev_x) < 1e-7:
            break
        prev_x = x.clone()
        prev_alpha = alpha.clone()
        

    return (x_trajectory, y_trajectory, track_norm_d, track_alpha, track_distance_PS, start_hitting_iterations, stop_hitting_iterations)

def run_experiment():
    os.makedirs("experiment results", exist_ok=True)  # create directory

    for problem in problems:
        problem_folder_path = os.path.join("experiment results", problem.name)
        os.makedirs(problem_folder_path, exist_ok=True)
        for seed in seeds:
            seed_folder_path = os.path.join(problem_folder_path, f"seed_{seed}")
            os.makedirs(seed_folder_path, exist_ok=True)
            data_folder_path = os.path.join(seed_folder_path, "data")
            os.makedirs(data_folder_path, exist_ok=True)
            plots_folder_path = os.path.join(seed_folder_path, "plots")
            os.makedirs(plots_folder_path, exist_ok=True)

            # Collect results for all aggregators
            results = {}
            for aggregator in aggregators:
                x_traj_list, y_traj_list, track_norm_d, track_alpha, track_distance_PS, start_hitting, stop_hitting = iterate(problem, aggregator, seed)
                x_trajectory = np.array(x_traj_list)
                y_trajectory = np.array(y_traj_list)
                track_norm_d = np.array(track_norm_d)
                track_alpha = np.array(track_alpha)
                track_distance_PS = np.array(track_distance_PS)
                start_hitting = np.array(start_hitting)
                stop_hitting = np.array(stop_hitting)
                results[aggregator.name] = {
                    'x_trajectory': x_trajectory,
                    'y_trajectory': y_trajectory,
                    'track_norm_d': track_norm_d,
                    'track_alpha': track_alpha,
                    'track_distance_PS': track_distance_PS,
                    'start_hitting_iterations': start_hitting,
                    'stop_hitting_iterations': stop_hitting
                }

                # Save data

                # Express start_hitting_iterations and stop_hitting_iterations as a list of Booleans
                start_hitting_iterations_bool = [False] * len(track_norm_d)
                stop_hitting_iterations_bool = [False] * len(track_norm_d)
                for index in start_hitting:
                    start_hitting_iterations_bool[index] = True 
                for index in stop_hitting:
                    stop_hitting_iterations_bool[index] = True 
                n_iteration = np.arange(len(track_norm_d))
                df = pd.DataFrame({
                    "n_iteration": n_iteration,
                    "x_trajectory": list(x_trajectory),
                    "y_trajectory": list(y_trajectory),
                    "norm_d": track_norm_d,
                    "alpha": track_alpha,
                    "measure_PS": track_distance_PS,
                    "Start_hitting_boundary": start_hitting_iterations_bool,
                    "stop_hitting_boundary": stop_hitting_iterations_bool
                })
                csv_filename = os.path.join(data_folder_path, f"{aggregator.name}.csv")
                pickle_filename = os.path.join(data_folder_path, f"{aggregator.name}.pkl")
                df.to_csv(csv_filename)
                df.to_pickle(pickle_filename)

            # Define aggregator categories and colors
            standard_aggregators = [agg for agg in results if not agg.endswith('*')]
            normalized_aggregators = [agg for agg in results if agg.endswith('*')]

            # Helper function to get plot data
            def get_values(data, key):
                if key == 'f1':
                    return data['y_trajectory'][:, 0]
                elif key == 'f2':
                    return data['y_trajectory'][:, 1]
                elif key == 'norm_d':
                    return data['track_norm_d']
                elif key == 'distance_PS':
                    return data['track_distance_PS']

            # Plot combined figure for standard aggregators (combined.png)
            fig_unnorm, axes_unnorm = plt.subplots(2, 2, figsize=(12, 12))
            for agg in standard_aggregators + normalized_aggregators:
                data = results[agg]
                n_iteration = np.arange(len(data['track_norm_d']))
                f1 = get_values(data, 'f1')
                f2 = get_values(data, 'f2')
                norm_d = get_values(data, 'norm_d')
                distance_PS = get_values(data, 'distance_PS')
                
                axes_unnorm[0, 0].plot(n_iteration, f1,  label=agg)
                axes_unnorm[0, 1].plot(n_iteration, f2,  label=agg)
                axes_unnorm[1, 0].plot(n_iteration, norm_d,  label=agg)
                axes_unnorm[1, 1].plot(n_iteration, distance_PS,  label=agg)
                if data['start_hitting_iterations'].size > 0:
                    start_idx = data['start_hitting_iterations']
                    axes_unnorm[0, 0].plot(start_idx, f1[start_idx], 'ro')
                    axes_unnorm[0, 1].plot(start_idx, f2[start_idx], 'ro')
                    axes_unnorm[1, 0].plot(start_idx, norm_d[start_idx], 'ro')
                    axes_unnorm[1, 1].plot(start_idx, distance_PS[start_idx], 'ro')
                if data['stop_hitting_iterations'].size > 0:
                    stop_idx = data['stop_hitting_iterations']
                    axes_unnorm[0, 0].plot(stop_idx, f1[stop_idx], 'go')
                    axes_unnorm[0, 1].plot(stop_idx, f2[stop_idx], 'go')
                    axes_unnorm[1, 0].plot(stop_idx, norm_d[stop_idx], 'go')
                    axes_unnorm[1, 1].plot(stop_idx, distance_PS[stop_idx], 'go')
            axes_unnorm[0, 0].set_title("First Loss Function")
            axes_unnorm[0, 1].set_title("Second Loss Function")
            axes_unnorm[1, 0].set_title("Norm of d")
            axes_unnorm[1, 1].set_title("Measure of Pareto Stationarity")
            for ax in axes_unnorm.flat:
                ax.legend()
            fig_unnorm.suptitle(f"Results for {problem.name}, seed {seed}")
            fig_unnorm.savefig(os.path.join(plots_folder_path, "combined.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_unnorm)

            # Plot comparison figure (combined2.png)
            fig_combined2, axes_combined2 = plt.subplots(2, 4, figsize=(24, 12))
            for agg in standard_aggregators:
                data = results[agg]
                n_iteration = np.arange(len(data['track_norm_d']))
                f1 = get_values(data, 'f1')
                f2 = get_values(data, 'f2')
                norm_d = get_values(data, 'norm_d')
                distance_PS = get_values(data, 'distance_PS')
                
                axes_combined2[0, 0].plot(n_iteration, f1,  label=agg)
                axes_combined2[0, 1].plot(n_iteration, f2,  label=agg)
                axes_combined2[0, 2].plot(n_iteration, norm_d,  label=agg)
                axes_combined2[0, 3].plot(n_iteration, distance_PS,  label=agg)
            
            # MGDA is also considered a starred plot.
            if MGDA in aggregators:
                normalized_aggregators = [MGDA.name] + normalized_aggregators

            for agg in normalized_aggregators:
                data = results[agg]
                n_iteration = np.arange(len(data['track_norm_d']))
                f1 = get_values(data, 'f1')
                f2 = get_values(data, 'f2')
                norm_d = get_values(data, 'norm_d')
                distance_PS = get_values(data, 'distance_PS')
                
                axes_combined2[1, 0].plot(n_iteration, f1,  label=agg)
                axes_combined2[1, 1].plot(n_iteration, f2,  label=agg)
                axes_combined2[1, 2].plot(n_iteration, norm_d,  label=agg)
                axes_combined2[1, 3].plot(n_iteration, distance_PS,  label=agg)
            titles = ["First Loss Function", "Second Loss Function", "Norm of d", "Measure of Pareto Stationarity"]
            for col, title in enumerate(titles):
                axes_combined2[0, col].set_title(f"{title} (Unnormalized)")
                axes_combined2[1, col].set_title(f"{title} (Normalized)")
                axes_combined2[0, col].legend()
                axes_combined2[1, col].legend()
            fig_combined2.suptitle(f"Results for {problem.name}, seed {seed}")
            fig_combined2.savefig(os.path.join(plots_folder_path, "combined2.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_combined2)

            # Plot individual plots for standard aggregators
            for plot_type, data_key in [("First_Loss_Function_Combined", 'f1'), ("Second_Loss_Function_Combined", 'f2'),
                                        ("Norm_of_d_Combined", 'norm_d'), ("Measure_of_Pareto_Stationarity_Combined", 'distance_PS')]:
                fig, ax = plt.subplots(figsize=(6, 6))
                for agg in standard_aggregators:
                    data = results[agg]
                    n_iteration = np.arange(len(data['track_norm_d']))
                    values = get_values(data, data_key)
                    
                    ax.plot(n_iteration, values,  label=agg)
                    if data['start_hitting_iterations'].size > 0:
                        start_idx = data['start_hitting_iterations']
                        ax.plot(start_idx, values[start_idx], 'ro')
                    if data['stop_hitting_iterations'].size > 0:
                        stop_idx = data['stop_hitting_iterations']
                        ax.plot(stop_idx, values[stop_idx], 'go')
                ax.set_xlabel("Iterations")
                ax.set_ylabel(plot_type.replace('_', ' '))
                ax.set_title(f"{plot_type.replace('_', ' ')} for {problem.name}, seed {seed}")
                ax.legend()
                fig.savefig(os.path.join(plots_folder_path, f"{plot_type}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

            # Plot individual normalized and unnormalized plots
            for norm_str, agg_list in [("Unnormalized", standard_aggregators), ("Normalized", normalized_aggregators)]:
                for plot_type, data_key in [("First_Loss_Function", 'f1'), ("Second_Loss_Function", 'f2'),
                                            ("Norm_of_d", 'norm_d'), ("Measure_of_Pareto_Stationarity", 'distance_PS')]:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    for agg in agg_list:
                        data = results[agg]
                        n_iteration = np.arange(len(data['track_norm_d']))
                        values = get_values(data, data_key)
                        
                        ax.plot(n_iteration, values,  label=agg)
                        if data['start_hitting_iterations'].size > 0:
                            start_idx = data['start_hitting_iterations']
                            ax.plot(start_idx, values[start_idx], 'ro')
                        if data['stop_hitting_iterations'].size > 0:
                            stop_idx = data['stop_hitting_iterations']
                            ax.plot(stop_idx, values[stop_idx], 'go')
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel(plot_type.replace('_', ' '))
                    ax.set_title(f"{plot_type.replace('_', ' ')} ({norm_str}) for {problem.name}, seed {seed}")
                    ax.legend()
                    fig.savefig(os.path.join(plots_folder_path, f"{plot_type}_{norm_str}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)

if __name__ == "__main__":
    run_experiment()
