import numpy as np
from matplotlib import pyplot as plt
from aggregators import MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star
import os
import pandas as pd
from synthetic_problems import zdt3, vlmop2, omnitest, ewq

problems = [vlmop2, omnitest, zdt3]
aggregators = [MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star]
seeds = [24, 42, 48, 100, 123]
seeds.sort()

def plot_from_data():
    for problem in problems:
        problem_folder_path = os.path.join("experiment results", problem.name)
        for seed in seeds:
            seed_folder_path = os.path.join(problem_folder_path, f"seed_{seed}")
            data_folder_path = os.path.join(seed_folder_path, "data")
            plots_folder_path = os.path.join(seed_folder_path, "plots")
            os.makedirs(plots_folder_path, exist_ok=True)

            # Load results for all aggregators
            results = {}
            for aggregator in aggregators:
                pickle_filename = os.path.join(data_folder_path, f"{aggregator.name}.pkl")
                df = pd.read_pickle(pickle_filename)
                x_trajectory = np.array(df['x_trajectory'].tolist())
                y_trajectory = np.array(df['y_trajectory'].tolist())
                track_norm_d = df['norm_d'].to_numpy()
                track_alpha = df['alpha'].to_numpy()
                track_distance_PS = df['measure_PS'].to_numpy()
                start_hitting = np.where(df['Start_hitting_boundary'])[0]
                stop_hitting = np.where(df['stop_hitting_boundary'])[0]
                results[aggregator.name] = {
                    'x_trajectory': x_trajectory,
                    'y_trajectory': y_trajectory,
                    'track_norm_d': track_norm_d,
                    'track_alpha': track_alpha,
                    'track_distance_PS': track_distance_PS,
                    'start_hitting_iterations': start_hitting,
                    'stop_hitting_iterations': stop_hitting
                }

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
            axes_unnorm[1, 0].set_title(r"$\|d\|$")
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
            titles = ["First Loss Function", "Second Loss Function", r"$\|d\|$", "Measure of Pareto Stationarity"]
            for col, title in enumerate(titles):
                axes_combined2[0, col].set_title(f"{title}")
                axes_combined2[1, col].set_title(f"{title}")
                axes_combined2[0, col].legend()
                axes_combined2[1, col].legend()
            fig_combined2.suptitle(f"Results for {problem.name}, seed {seed}")
            fig_combined2.savefig(os.path.join(plots_folder_path, "combined2.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_combined2)

            # Plot individual plots for standard aggregators
            for plot_type, data_key in [("First_Loss_Function", 'f1'), ("Second_Loss_Function", 'f2'),
                                        ("Norm_of_d", 'norm_d'), ("Measure_of_Pareto_Stationarity", 'distance_PS')]:
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
                if plot_type == "Norm_of_d":
                    ax.set_title(r"$\|d\|$")
                else:
                    ax.set_title(plot_type.replace('_', ' '))
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
                    if plot_type == "Norm_of_d":
                        ax.set_title(r"$\|d\|$")
                    else:
                        ax.set_title(plot_type.replace('_', ' '))
                    ax.legend()
                    fig.savefig(os.path.join(plots_folder_path, f"{plot_type}_{norm_str}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)

if __name__ == "__main__":
    plot_from_data()