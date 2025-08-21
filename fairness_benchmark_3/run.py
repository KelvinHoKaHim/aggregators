from libmoon.util import mtl
from typing import List
from libmoon.problem.mtl.objectives import BinaryCrossEntropyLoss, DEOHyperbolicTangentRelaxation, DEOHyperbolicTangentRelaxation2, DEOEmpirical, DEOEmpirical2
import torch 
import os 
import numpy as np 
import pandas as pd
from utils.aggregators import MGDA, UPGrad, UPGrad_star, DualProj, DualProj_star, Nash_MTL, Nash_MTL_star, aggregator_dict
import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
path = "fairness_classification_3"
os.makedirs("fairness_classification", exist_ok=True) # create a directory to store model and data
data_train = mtl.get_dataset("adult", type = "train")
data_test = mtl.get_dataset("adult", type = "test")
data_val = mtl.get_dataset("adult", type = "val")


# x represent the feature, y the label, and s is the sensible attribute (sensible attrbute is defined to be 'sex' by default)


def evaluate(model):
    DEO = DEOEmpirical()
    DEO2 = DEOEmpirical2()
    x = data_test.x.to(device) 
    y = data_test.y.to(device)
    s = data_test.s1.to(device)
    prediction = model(x)["logits"] # Don't know why LibMoon defined their foward function like this. 
                                    # The forward function returns a dictionary, in which the only key 
                                    # is "logists", and the corresponding value is the model prediction.
    n_data = len(prediction)
    success_rate = len(prediction[torch.squeeze(prediction > 0) == y.bool()]) / n_data # a value between 0 and 1, the higher, the better
    fairness1 = DEO(prediction, labels = y, sensible_attribute = s).item() # a positive value, the smaller, the better
    fairness2 = DEO2(prediction, labels = y, sensible_attribute = s).item() 
    return (success_rate, fairness1, fairness2)
                                    
def train(aggregator, seed : int, epochs : int, learning_rate : float, eps : float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = mtl.model_from_dataset("adult", architecture="M4")   # a fully connected NN
    model = model.to(device)

    curr_aggregator = aggregator() # initialised the aggregator
    mgda = MGDA() # initialised the MGDA
    criterion1 = BinaryCrossEntropyLoss() 
    criterion2 = DEOHyperbolicTangentRelaxation()
    criterion3  = DEOHyperbolicTangentRelaxation2()
    track_loss1 = []
    track_loss2 = []
    track_loss3 = []
    track_d = []
    track_alpha = []
    track_d_MGDA = []

    x = data_train.x.to(device)
    y = data_train.y.to(device)
    s = data_train.s1.to(device)
    prev_alpha = torch.ones(3).to(device)  # Initialize prev_alpha

    try:
        for epoch in range(epochs): # start training
            model.zero_grad()
            prediction = model(x)["logits"]

            # Compute two losses
            loss1 = criterion1(prediction, labels = y)
            loss2 = criterion2(prediction, labels = y, sensible_attribute = s)
            loss3 = criterion3(prediction, labels = y, sensible_attribute = s)

            # Find gradient for loss1
            loss1.backward(retain_graph=True)
            grad1 = [p.grad.clone() for p in model.parameters()]
            grad1_flat = torch.cat([g.view(-1) for g in grad1])  # Flatten into a vector
            model.zero_grad()

            # Compute gradients for loss2
            loss2.backward(retain_graph=True)
            grad2 = [p.grad.clone() for p in model.parameters()]
            grad2_flat = torch.cat([g.view(-1) for g in grad2])  # Flatten into a vector
            model.zero_grad()

            # Compute gradients for loss3
            loss3.backward(retain_graph=True)
            grad3 = [p.grad.clone() for p in model.parameters()]
            grad3_flat = torch.cat([g.view(-1) for g in grad3])  # Flatten into a vector
            model.zero_grad()
    
            jacobian = torch.stack([grad1_flat, grad2_flat, grad3_flat], dim=0) # It has shape 2xn where n is the total number of model parameters
            if aggregator.name == "Nash-MTL" or aggregator.name == "Nash-MTL*":
                d, alpha = curr_aggregator(jacobian, prev_alpha)
                prev_alpha = alpha
            else:
                d, alpha = curr_aggregator(jacobian)
            d_mgda, alpha_mgda = mgda(jacobian)
            total_loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3
            total_loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= learning_rate * p.grad

            norm_d = torch.norm(d).item()
            norm_d_mgda = torch.norm(d_mgda).item()
            if (epoch + 1) % 10 == 0:
                print(f"Training with {aggregator.name} with seed {seed}")
                print(f"Epoch {epoch+1}/{epochs}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}, d_MGDA = {norm_d_mgda}")
            track_loss1.append(loss1.item())
            track_loss2.append(loss2.item())
            track_loss3.append(loss3.item())
            track_alpha.append(alpha.detach().cpu().numpy())
            track_d.append(norm_d)
            track_d_MGDA.append(norm_d_mgda)
            if norm_d_mgda < eps or norm_d < eps:
                print(f"EARLY STOPPING at epoch {epoch+1}/{epochs}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}")
                break

        log = f"{aggregator.name}: Training ended at eppoch {epoch+1}/{epochs}. Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}, d_MGDA = {norm_d_mgda}"
    except Exception as e:
        e = str(e)
        track_loss1 = [-1 for i in range(epochs)]
        track_d = [-1 for i in range(epochs)]
        track_alpha = [-1 for i in range(epochs)]
        track_d_MGDA = [-1 for i in range(epochs)]
        log = f"{aggregator.name}: Training interrupted at {epoch+1}/{epochs} with message '{e}'. Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}, d_MGDA = {norm_d_mgda}"

    return (model, track_loss1, track_loss2, track_loss3, track_d, track_alpha, track_d_MGDA, log)


def run_experiment(aggregators : List, seeds : List[int], lr : float, epochs : int, eps : float):
    results_path = "fairness_benchmark_3/results"
    # print experiment details
    print(20*"=" + "Experiment Details" + 20*"=")
    print(f"Aggregators :{[aggregator.name for aggregator in aggregators]}")
    print(f"Dataset :Adult")
    print(f"Seeds :{seeds}")
    print(f"Learing rate :{lr}")
    print(f"Number of epochs :{epochs}")
    print(f"Tolorance :{eps}")
    print(58*"=")

    os.makedirs(results_path, exist_ok=True)  

    # record experiment parameters in a txt file
    with open(f"{results_path}/experiment_details.txt", "w") as f: 
        print(20*"=" + "Experiment Details" + 20*"=", file = f)
        print(f"Aggregators :{[aggregator.name for aggregator in aggregators]}", file = f)
        print(f"Dataset :Adult", file = f)
        print(f"Seeds :{seeds}", file = f)
        print(f"Learing rate :{lr}", file = f)
        print(f"Number of epoch :{epochs}", file = f)
        print(f"Tolorance :{eps}", file = f)
        print(58*"=", file = f)

    for seed in seeds:
        seed_folder_path = os.path.join(results_path, f"seed_{seed}")
        os.makedirs(seed_folder_path, exist_ok=True)
        data_folder_path = os.path.join(seed_folder_path, "data")
        os.makedirs(data_folder_path, exist_ok=True)

        # Collect results for all aggregators
        for aggregator in aggregators:
            print(40*"=")
            print("Now testing with...")
            print(f"Aggregator : {aggregator.name}")
            print(f"Seed : {seed}")

            model, track_loss1, track_loss2, track_loss3, track_norm_d, track_alpha, track_measure_to_PS, log = train(aggregator, seed, epochs, lr, eps)

            with open(os.path.join(seed_folder_path, "log.txt"), "a") as f:
                f.write(log + "\n")

            model_path = os.path.join(data_folder_path, f"{aggregator.name}.pt")
            torch.save(model.state_dict(), model_path) # save model

            n_iteration = np.arange(len(track_norm_d))
            df = pd.DataFrame({
                "n_iteration": n_iteration,
                "cross_entropy" : track_loss1,
                "deo1" : track_loss2,
                "deo2" : track_loss3,
                "norm_d": track_norm_d,
                "alpha": track_alpha,
                "measure_PS": track_measure_to_PS,
            })
            csv_filename = os.path.join(data_folder_path, f"{aggregator.name}.csv")
            pickle_filename = os.path.join(data_folder_path, f"{aggregator.name}.pkl")
            df.to_csv(csv_filename)
            df.to_pickle(pickle_filename)

            success_rate, fairness1, fairness2 = evaluate(model)
            print(f"Success rate = {success_rate}, Fairness1 = {fairness1}, Fariness2 = {fairness2}")
            eval_filename = f"{aggregator.name}_evaluation.txt"
            np.savetxt(os.path.join(data_folder_path, eval_filename), [success_rate, fairness1, fairness2])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameters for running fairness(three objectives) benchmark")

    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=400000, help="Number of epochs")
    parser.add_argument("--eps", type=float, default=1e-3, help="Stopping criteria tolorance")
    parser.add_argument(
        "--aggregators", 
        type=str, 
        nargs="+",
        default=["MGDA", "UPGrad", "UPGrad*", "DualProj", "DualProj*","Nash-MTL", "Nash-MTL*"],  # Default list of aggregators
        help="List of aggregators to be used in experiment"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+",
        default=[24, 42, 64, 100],  # Default random seeds
        help="List of randoms seeds"
    )

    args = parser.parse_args()

    aggregators = [aggregator_dict[aggregators_name] for aggregators_name in args.aggregators if aggregators_name in aggregator_dict]
    seeds = args.seeds
    lr = args.lr 
    epochs = args.epochs
    eps = args.eps

    run_experiment(aggregators, seeds, lr, epochs, eps)
            



        
