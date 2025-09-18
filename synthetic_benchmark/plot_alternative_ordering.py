import numpy as np
from matplotlib import pyplot as plt
from utils.aggregators import MGDA, Nash_MTL, Nash_MTL_star, UPGrad, UPGrad_star, DualProj, DualProj_star, aggregator_dict
import os
import pandas as pd
from typing import Set
from synthetic_benchmark.run import aggregator_dict
import argparse

# When curves are ugly, we artificilly clip the curves. Hard code clipping threshold when necessary. Set to np.inf by dafult (i.e. no need clipping)
x_clip = {
    'MGDA': np.inf,
    'Nash-MTL': np.inf,
    'Nash-MTL*': np.inf,
    'UPGrad': np.inf,
    'UPGrad*': np.inf,
    'DualProj': np.inf,
    'DualProj*': np.inf
}

colours = {
    'MGDA': 'blue',
    'Nash-MTL': 'gold',
    'Nash-MTL*': 'orange',
    'UPGrad': 'green',
    'UPGrad*': 'olive',
    'DualProj': 'purple',
    'DualProj*': 'mediumvioletred'
}

def plot_experiment_results(only_plot_aggregator : Set[int], only_plot_seed : Set[int], image_format : str):

    results_path = "synthetic_benchmark/results"
    if not os.path.isdir(results_path):
        print(f"Warning : Directory {results_path} does not exist")
        return 
    
    problem_list = os.listdir(results_path)
    for problem in problem_list:
        problem_path = os.path.join(results_path, problem)
        if not os.path.isdir(problem_path):
            continue
        seed_list = os.listdir(problem_path)
        for seed_directory in seed_list:
            seed_path = os.path.join(problem_path, seed_directory)
            if not (os.path.isdir(seed_path) and (lambda s : s[0] == "seed" and s[1].isdigit())(seed_directory.split("_"))): # pass if directory is not named in the format "seed_<seed_number>"
                continue
            seed = seed_directory.split("_")[1]
            if len(only_plot_seed) > 0 and seed not in only_plot_seed: 
                continue
            data_path = os.path.join(seed_path, "data")
            if not os.path.isdir(data_path):
                print(f"Warning : Directory {data_path} does not exist")
                continue 
            plot_path = os.path.join(seed_path, "plots_alternative_ordering")
            os.makedirs(plot_path, exist_ok=True)
            #aggregator_list =  sorted(os.listdir(data_path))
            aggregator_list = ["MGDA.pkl", "Nash-MTL.pkl", "Nash-MTL*.pkl", "UPGrad.pkl", "UPGrad*.pkl", "DualProj.pkl", "DualProj*.pkl"]

            fig_combined, ax_combined = plt.subplots(2, 2, figsize=(12, 12))
            fig_combined2, ax_combined2 = plt.subplots(2, 4, figsize=(24, 12))
            fig_f1, ax_f1 = plt.subplots(figsize=(6, 6)) # plot for first loss function
            fig_f2, ax_f2 = plt.subplots(figsize=(6, 6)) # plot for second loss function
            fig_d, ax_d = plt.subplots(figsize=(6, 6)) # plot for the norm of d
            fig_dps, ax_dps = plt.subplots(figsize=(6, 6)) # plot for the measure to Pareto stationarity

            for data_filename in aggregator_list:
                if not (lambda s : s[0] in aggregator_dict and s[1] == "pkl")(data_filename.split(".")): # pass if file name is not of the format "<aggregator_name>.pkl", where aggregator_name must be among MGDA, Nash-MTL, Nash-MTL*, UPGrad, UPGrad*, DualProj, DualProj*
                    continue
                aggregator_name = data_filename.split(".")[0]
                if len(only_plot_aggregator) > 0 and aggregator_name not in only_plot_aggregator:
                    continue
                pickle_file = os.path.join(data_path, data_filename)
                print(f"Plotting from {pickle_file}")
                df = pd.read_pickle(pickle_file)
                x_trajectory = np.array(df['x_trajectory'].tolist())
                y_trajectory = np.array(df['y_trajectory'].tolist())
                track_norm_d = df['norm_d'].to_numpy()
                track_alpha = df['alpha'].to_numpy()
                track_measure_to_PS = df['measure_PS'].to_numpy()
                f1, f2 = y_trajectory.T

                clipping_threshold = x_clip[aggregator_name]
                n_iteration = np.arange(int(np.min([len(track_norm_d), clipping_threshold])))

                # plot individual figures
                ax_f1.plot(n_iteration, f1, color = colours[aggregator_name],label = aggregator_name)
                ax_f2.plot(n_iteration, f2, color = colours[aggregator_name],label = aggregator_name)
                ax_d.plot(n_iteration, track_norm_d, color = colours[aggregator_name],label = aggregator_name)
                ax_dps.plot(n_iteration, track_measure_to_PS, color = colours[aggregator_name],label = aggregator_name)
            

                # plot combined.png 
                ax_combined[0,0].plot(n_iteration, f1, color = colours[aggregator_name],label = aggregator_name)
                ax_combined[0,1].plot(n_iteration, f2, color = colours[aggregator_name],label = aggregator_name)
                ax_combined[1,0].plot(n_iteration, track_norm_d, color = colours[aggregator_name],label = aggregator_name)
                ax_combined[1,1].plot(n_iteration, track_measure_to_PS, color = colours[aggregator_name],label = aggregator_name)

                # plt combined2.png
                row_number = 1 if aggregator_name.endswith("*") else 0 # 
                ax_combined2[row_number,0].plot(n_iteration, f1, color = colours[aggregator_name],label = aggregator_name)
                ax_combined2[row_number,1].plot(n_iteration, f2, color = colours[aggregator_name],label = aggregator_name)
                ax_combined2[row_number,2].plot(n_iteration, track_norm_d, color = colours[aggregator_name],label = aggregator_name)
                ax_combined2[row_number,3].plot(n_iteration, track_measure_to_PS, color = colours[aggregator_name],label = aggregator_name)

                if aggregator_name == "MGDA": # MGDA is the only aggregator that is plotted on both rows
                    ax_combined2[1,0].plot(n_iteration, f1, color = colours[aggregator_name],label = aggregator_name)
                    ax_combined2[1,1].plot(n_iteration, f2, color = colours[aggregator_name],label = aggregator_name)
                    ax_combined2[1,2].plot(n_iteration, track_norm_d, color = colours[aggregator_name],label = aggregator_name)
                    ax_combined2[1,3].plot(n_iteration, track_measure_to_PS, color = colours[aggregator_name],label = aggregator_name)



            ax_f1.set_title("First loss function", fontsize=18)
            ax_f2.set_title("Second loss function", fontsize=18)
            ax_d.set_title(r"$\|d\|$", fontsize=18)
            ax_dps.set_title("Measure of Pareto stationarity", fontsize=18)

            ax_f1.legend(loc='upper right', prop={'size': 14})
            ax_f2.legend(loc='upper right', prop={'size': 14})
            ax_d.legend(loc='upper right', prop={'size': 14})
            ax_dps.legend(loc='upper right', prop={'size': 14})

            fig_f1.savefig(os.path.join(plot_path, f"First_loss_function.{image_format}"), dpi = 300)
            fig_f2.savefig(os.path.join(plot_path, f"Second_loss_function.{image_format}"), dpi = 300)
            fig_d.savefig(os.path.join(plot_path, f"d.{image_format}"), dpi = 300)
            fig_dps.savefig(os.path.join(plot_path, f"Measure_of_Pareto_stationarity.{image_format}"), dpi = 300)

            plt.close(fig_f1)
            plt.close(fig_f2)
            plt.close(fig_d)
            plt.close(fig_dps)

            ax_combined[0,0].set_title("First loss function", fontsize=18)
            ax_combined[0,1].set_title("Second loss function", fontsize=18)
            ax_combined[1,0].set_title(r"$\|d\|$", fontsize=18)
            ax_combined[1,1].set_title("Measure of Pareto stationarity", fontsize=18)

            ax_combined[0,0].legend(loc='upper right', prop={'size': 14})
            ax_combined[0,1].legend(loc='upper right', prop={'size': 14})
            ax_combined[1,0].legend(loc='upper right', prop={'size': 14})
            ax_combined[1,1].legend(loc='upper right', prop={'size': 14})

            fig_combined.savefig(os.path.join(plot_path, f"Combined.{image_format}"), dpi = 300)
            plt.close(fig_combined)

            ax_combined2[0,0].set_title("First loss function", fontsize=18)
            ax_combined2[0,1].set_title("Second loss function", fontsize=18)
            ax_combined2[0,2].set_title(r"$\|d\|$", fontsize=18)
            ax_combined2[0,3].set_title("Measure of Pareto stationarity", fontsize=18)

            ax_combined2[0,0].legend(loc='upper right', prop={'size': 14})
            ax_combined2[0,1].legend(loc='upper right', prop={'size': 14})
            ax_combined2[0,2].legend(loc='upper right', prop={'size': 14})
            ax_combined2[0,3].legend(loc='upper right', prop={'size': 14})

            ax_combined2[1,0].set_title("First loss function", fontsize=18)
            ax_combined2[1,1].set_title("Second loss function", fontsize=18)
            ax_combined2[1,2].set_title(r"$\|d\|$", fontsize=18)
            ax_combined2[1,3].set_title("Measure of Pareto stationarity", fontsize=18)

            ax_combined2[1,0].legend(loc='upper right', prop={'size': 14})
            ax_combined2[1,1].legend(loc='upper right', prop={'size': 14})
            ax_combined2[1,2].legend(loc='upper right', prop={'size': 14})
            ax_combined2[1,3].legend(loc='upper right', prop={'size': 14})

            fig_combined2.savefig(os.path.join(plot_path, f"Combined2.{image_format}"), dpi = 300)
            plt.close(fig_combined2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for plotting graphs for synthetic problem benchmark")

    parser.add_argument("--format", type=str, default="png", help="Format of image files")
    parser.add_argument(
        "--only_plot_seeds", 
        type=str, 
        nargs="+",
        default=[],
        help="If provided, then plot only the provided seeds"
    )
    parser.add_argument(
        "--only_plot_aggregators", 
        type=str, 
        nargs="+",
        default=[],
        help="If provided, then plot only the provided seeds"
    )

    args = parser.parse_args()
    only_plot_aggregators = args.only_plot_aggregators
    only_plot_seeds = args.only_plot_seeds
    image_format = args.format

    plot_experiment_results(only_plot_aggregators, only_plot_seeds, image_format)