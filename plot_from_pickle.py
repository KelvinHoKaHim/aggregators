import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define constants
problems = ["omnitest", "vlmop2", "zdt3"]
seeds = [24, 42, 48, 100, 123]
aggregators = ["MGDA", "UPGrad", "DualProj", "NASH_MTL"]
aggregator_colors = {"MGDA": 'blue', "UPGrad": 'green', "DualProj": 'red', "NASH_MTL": "orange"}
aggregator_colors_normalized = {"MGDA": 'blue', "UPGrad": 'lime', "DualProj": 'magenta', "NASH_MTL": "gold"}
eta = 0.001  # Hardcoded as per original exp.py

for problem in problems:
    for seed in seeds:
        print(f"Plotting for problem {problem}, seed {seed}")
        seed_folder = os.path.join("experiment results", problem, f"seed_{seed}")
        if not os.path.exists(seed_folder):
            print(f"Warning: {seed_folder} does not exist. Skipping.")
            continue

        # Load data from pickle files
        results = {}
        for aggregator in aggregators:
            results[aggregator] = {}
            for normalize_d in [False, True]:
                norm_str = 'normalized' if normalize_d else 'unnormalized'
                pickle_filename = f"{aggregator}_{norm_str}.pkl"
                pickle_path = os.path.join(seed_folder, pickle_filename)
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as f:
                        data = pickle.load(f)
                    y_trajectory = data['y_trajectory']
                    f1 = y_trajectory[:, 0]
                    f2 = y_trajectory[:, 1]
                    n_iteration = np.arange(len(f1))
                    track_norm = data['track_norm_d']
                    track_distance_PS = data['track_distance_PS']
                    start_hitting = data['start_hitting_iterations']
                    stop_hitting = data['stop_hitting_iterations']
                    results[aggregator][normalize_d] = {
                        'n_iteration': n_iteration,
                        'f1': f1,
                        'f2': f2,
                        'track_norm': track_norm,
                        'track_distance_PS': track_distance_PS,
                        'start_hitting': start_hitting,
                        'stop_hitting': stop_hitting
                    }
                else:
                    print(f"Warning: {pickle_path} does not exist.")

        # Create combined.png (2x2 subplots)
        fig_unnorm, axes_unnorm = plt.subplots(2, 2, figsize=(12, 12))
        for aggregator in aggregators:
            for normalize_d in [False, True]:
                if aggregator != "MGDA" or normalize_d == False:
                    if normalize_d in results[aggregator]:
                        data = results[aggregator][normalize_d]
                        n_iteration = data['n_iteration']
                        f1 = data['f1']
                        f2 = data['f2']
                        track_norm = data['track_norm']
                        track_distance_PS = data['track_distance_PS']
                        start_hitting = data['start_hitting']
                        stop_hitting = data['stop_hitting']
                        label = f"{aggregator}{'*' if (normalize_d and aggregator != 'MGDA') else ''}"
                        color = aggregator_colors_normalized[aggregator] if (normalize_d and aggregator != 'MGDA') else aggregator_colors[aggregator]
                        # Plot on axes_unnorm
                        for ax, values in zip(
                            axes_unnorm.flat,
                            [f1, f2, track_norm, track_distance_PS]
                        ):
                            ax.plot(n_iteration, values, color=color, linestyle='-', label=label)
                            if start_hitting:
                                ax.plot(start_hitting, values[start_hitting], 'ro')
                            if stop_hitting:
                                ax.plot(stop_hitting, values[stop_hitting], 'go')
        axes_unnorm[0, 0].set_title("First Loss Function")
        axes_unnorm[0, 1].set_title("Second Loss Function")
        axes_unnorm[1, 0].set_title("Norm of d")
        axes_unnorm[1, 1].set_title("Measure of Pareto stationarity")
        for ax in axes_unnorm.flat:
            ax.legend()
        fig_unnorm.suptitle(f"Experiment results for {problem} with seed {seed}, eta={eta}")
        fig_unnorm.savefig(os.path.join(seed_folder, "combined.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_unnorm)

        # Create combined2.png (2x4 subplots)
        fig_combined2, axes_combined2 = plt.subplots(2, 4, figsize=(24, 12))
        for row, normalize_d in enumerate([False, True]):
            for aggregator in aggregators:
                if normalize_d in results[aggregator]:
                    data = results[aggregator][normalize_d]
                    n_iteration = data['n_iteration']
                    f1 = data['f1']
                    f2 = data['f2']
                    track_norm = data['track_norm']
                    track_distance_PS = data['track_distance_PS']
                    start_hitting = data['start_hitting']
                    stop_hitting = data['stop_hitting']
                    label = f"{aggregator}{'*' if normalize_d else ''}"
                    color = aggregator_colors_normalized[aggregator] if normalize_d else aggregator_colors[aggregator]
                    # Plot on axes_combined2
                    for col, values in enumerate([f1, f2, track_norm, track_distance_PS]):
                        ax = axes_combined2[row, col]
                        ax.plot(n_iteration, values, color=color, linestyle='-', label=label)
                        if start_hitting:
                            ax.plot(start_hitting, values[start_hitting], 'ro')
                        if stop_hitting:
                            ax.plot(stop_hitting, values[stop_hitting], 'go')
                        ax.legend()
        titles = ["First Loss Function", "Second Loss Function", "Norm of d", "Measure of Pareto stationarity"]
        for col in range(4):
            axes_combined2[0, col].set_title(f"{titles[col]} (Unnormalized)")
            axes_combined2[1, col].set_title(f"{titles[col]} (Normalized)")
        fig_combined2.suptitle(f"Experiment results for {problem} with seed {seed}, eta={eta}")
        fig_combined2.savefig(os.path.join(seed_folder, "combined2.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_combined2)

        # Create individual combined plots
        for plot_type, data_key in [
            ("First_Loss_Function_Combined", 'f1'),
            ("Second_Loss_Function_Combined", 'f2'),
            ("Norm_of_d_Combined", 'track_norm'),
            ("Measure_of_Pareto_Stationarity_Combined", 'track_distance_PS')
        ]:
            fig, ax = plt.subplots(figsize=(6, 6))
            for aggregator in aggregators:
                for normalize_d in [False, True]:
                    if aggregator != "MGDA" or normalize_d == False:
                        if normalize_d in results[aggregator]:
                            data = results[aggregator][normalize_d]
                            n_iteration = data['n_iteration']
                            values = data[data_key]
                            start_hitting = data['start_hitting']
                            stop_hitting = data['stop_hitting']
                            label = f"{aggregator}{'*' if (normalize_d and aggregator != 'MGDA') else ''}"
                            color = aggregator_colors_normalized[aggregator] if (normalize_d and aggregator != 'MGDA') else aggregator_colors[aggregator]
                            ax.plot(n_iteration, values, color=color, label=label)
                            if start_hitting:
                                ax.plot(start_hitting, values[start_hitting], 'ro')
                            if stop_hitting:
                                ax.plot(stop_hitting, values[stop_hitting], 'go')
            ax.set_xlabel("Iterations")
            ax.set_ylabel(plot_type.replace('_', ' '))
            ax.set_title(f"{plot_type.replace('_', ' ')} for {problem} with seed {seed}")
            ax.legend()
            fig.savefig(os.path.join(seed_folder, f"{plot_type}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Create individual plots for unnormalized and normalized
        for norm_str, normalize_d in [("Unnormalized", False), ("Normalized", True)]:
            for plot_type, data_key in [
                ("First_Loss_Function", 'f1'),
                ("Second_Loss_Function", 'f2'),
                ("Norm_of_d", 'track_norm'),
                ("Measure_of_Pareto_Stationarity", 'track_distance_PS')
            ]:
                fig, ax = plt.subplots(figsize=(6, 6))
                for aggregator in aggregators:
                    if normalize_d in results[aggregator]:
                        data = results[aggregator][normalize_d]
                        n_iteration = data['n_iteration']
                        values = data[data_key]
                        start_hitting = data['start_hitting']
                        stop_hitting = data['stop_hitting']
                        label = f"{aggregator}{'*' if normalize_d else ''}"
                        color = aggregator_colors_normalized[aggregator] if normalize_d else aggregator_colors[aggregator]
                        ax.plot(n_iteration, values, color=color, label=label)
                        if start_hitting:
                            ax.plot(start_hitting, values[start_hitting], 'ro')
                        if stop_hitting:
                            ax.plot(stop_hitting, values[stop_hitting], 'go')
                ax.set_xlabel("Iterations")
                ax.set_ylabel(plot_type.replace('_', ' '))
                ax.set_title(f"{plot_type.replace('_', ' ')} ({norm_str}) for {problem} with seed {seed}")
                ax.legend()
                fig.savefig(os.path.join(seed_folder, f"{plot_type}_{norm_str}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

print("Plotting completed.")