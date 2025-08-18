from matplotlib import pyplot as plt 
import os 
import pandas as pd
import numpy as np 


file_path = "fairness_classification_3"
output_path = os.path.join(file_path, "plots")
os.makedirs(output_path, exist_ok=True)

def plot(seed = int):
    colours = {
    'MGDA': 'blue',
    'Nash-MTL': 'gold',
    'Nash-MTL*': 'orange',
    'UPGrad': 'green',
    'UPGrad*': 'olive',
    'DualProj': 'purple',
    'DualProj*': 'mediumvioletred'
}

    fig_combined, ax_combined = plt.subplots(2, 3, figsize = (18,12))
    fig_individual, ax_individual = plt.subplots(figsize = (6,6))
    fig_f1, ax_f1 = plt.subplots(figsize = (6,6))
    fig_f2, ax_f2 = plt.subplots(figsize = (6,6))
    fig_f3, ax_f3 = plt.subplots(figsize = (6,6))
    fig_d, ax_d = plt.subplots(figsize = (6,6))
    fig_dps, ax_dps = plt.subplots(figsize = (6,6))


    #dir_list = sorted(os.listdir(file_path))
    dir_list = ["MGDA", "Nash-MTL", "UPGrad", "DualProj", "Nash-MTL*", "UPGrad*", "DualProj*"]
    for aggregator in dir_list:
        curr_dir = os.path.join(file_path, aggregator)
        if os.path.isfile(curr_dir):
            continue 
        seed_list = os.listdir(curr_dir)
        for curr_seed in seed_list:
            if curr_seed.replace("_", "") != str(seed):
                continue 
            print(curr_seed, curr_seed.replace("_", "") != str(seed))
            curr_seed_path = os.path.join(curr_dir, curr_seed)
            csv_path = os.path.join(curr_seed_path, f"{aggregator}_data.csv")
            df = pd.read_csv(csv_path)

            colour = colours[aggregator]
            loss_1 = df["first loss function"].to_numpy()
            loss_2 = df['second loss function'].to_numpy()
            loss_3 = df['third loss function'].to_numpy()
            norm_d = df["norm of d"].to_numpy()
            d_ps = df["measure of Pareto stationarity"].to_numpy()
            n_iteration = np.arange(len(loss_1))

            ax_combined[0,0].plot(n_iteration, loss_1, color = colour, label = aggregator)
            ax_f1.plot(n_iteration, loss_1, color = colour, label = aggregator)
            ax_combined[0,0].set_title("First loss function")
            ax_f1.set_title("First loss function")

            ax_combined[0,1].plot(n_iteration, loss_2, color = colour, label = aggregator)
            ax_f2.plot(n_iteration, loss_2, color = colour, label = aggregator)
            ax_combined[0,1].set_title("Second loss function")
            ax_f2.set_title("Second loss function")

            ax_combined[0,2].plot(n_iteration, loss_3, color = colour, label = aggregator)
            ax_f3.plot(n_iteration, loss_3, color = colour, label = aggregator)
            ax_combined[0,1].set_title("Third loss function")
            ax_f3.set_title("Third loss function")

            ax_combined[1,0].plot(n_iteration, norm_d, color = colour, label = aggregator)
            ax_d.plot(n_iteration, norm_d, color = colour, label = aggregator)
            ax_combined[1,0].set_title(r"$\|d\|$")
            ax_d.set_title(r"$\|d\|$")

            ax_combined[1,1].plot(n_iteration, d_ps, color = colour, label = aggregator)
            ax_dps.plot(n_iteration, d_ps, color = colour, label = aggregator)
            ax_combined[1,1].set_title("Measure of Pareto Stationarity")
            ax_dps.set_title("Measure of Pareto Stationarity")


    ax_combined[0,0].legend()
    ax_combined[0,1].legend()
    ax_combined[0,2].legend()
    ax_combined[1,0].legend()
    ax_combined[1,1].legend()
    ax_f1.legend()
    ax_f2.legend()
    ax_f2.legend()
    ax_d.legend()
    ax_dps.legend()
    fig_combined.savefig(os.path.join(output_path, f"{str(seed)}_combined.png"), dpi = 300)
    plt.close(fig_combined)
    fig_f1.savefig(os.path.join(output_path, f"{str(seed)}_first_loss_function.png"), dpi = 300)
    plt.close(fig_f1)
    fig_f2.savefig(os.path.join(output_path, f"{str(seed)}_second_loss_function.png"), dpi = 300)
    plt.close(fig_f2)
    fig_f3.savefig(os.path.join(output_path, f"{str(seed)}_third_loss_function.png"), dpi = 300)
    plt.close(fig_f3)
    fig_d.savefig(os.path.join(output_path, f"{str(seed)}_norm_of_d.png"), dpi = 300)
    plt.close(fig_d)
    fig_dps.savefig(os.path.join(output_path, f"{str(seed)}_measure_of_Pareto_stationarity.png"), dpi = 300)
    plt.close(fig_dps)

            

        
        
if __name__ == "__main__":
    plot(64)