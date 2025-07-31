from libmoon.util import mtl
from libmoon.problem.mtl.objectives import BinaryCrossEntropyLoss, DEOHyperbolicTangentRelaxation
import torch 
import os 
import numpy as np 
import pandas as pd
import pickle
from mgda import MGDA
from upgrad import UPGrad
from matplotlib import pyplot as plt 

path = "fairness_classification"
os.makedirs("fairness_classification", exist_ok=True) # create a directory to store model and data

data_train = mtl.get_dataset("adult", type = "train")
data_test = mtl.get_dataset("adult", type = "test")
data_val = mtl.get_dataset("adult", type = "val")
model = mtl.model_from_dataset("adult", architecture="M4")   # a fully connected NN

# x represent the feature, y the label, and s is the sensible attribute (sensible attrbute is defined to be 'sex' by default)


def evaluate():
    DEO = DEOHyperbolicTangentRelaxation()
    x = data_test.x 
    y = data_test.y 
    s = data_test.s1
    prediction = model(x)["logits"] # Don't know why LibMoon defined their foward function like this. 
                                    # The forward function returns a dictionary, in which the only key 
                                    # is "logists", and the corresponding value is the model prediction.
    n_data = len(prediction)
    success_rate = len(prediction[torch.squeeze(prediction > 0) == y.bool()]) / n_data # a value between 0 and 1, the higher, the better
    fairness = DEO(prediction, labels = y, sensible_attribute = s).item() # a positive value, the smaller, the better
    return (success_rate, fairness)
                                    
def train(aggregator, mgda, eps = 1e-3, learning_rate = 0.01, num_epochs = 50000):
    criterion1 = BinaryCrossEntropyLoss()
    criterion2 = DEOHyperbolicTangentRelaxation()
    track_loss1 = []
    track_loss2 = []
    track_d = []
    track_d_MGDA = []

    x = data_train.x
    y = data_train.y
    s = data_train.s1

    for epoch in range(num_epochs): # start training
        model.zero_grad()
        prediction = model(x)["logits"]

        # Compute two losses
        loss1 = criterion1(prediction, labels = y)
        loss2 = criterion2(prediction, labels = y, sensible_attribute = s)

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
 
        jacobian = torch.stack([grad1_flat, grad2_flat], dim=0) # It has shape 2xn where n is the total number of model parameters
        d, alpha = aggregator(jacobian)
        d_mgda, alpha_mgda = mgda(jacobian)
        total_loss = alpha[0] * loss1 + alpha[1] * loss2
        total_loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= learning_rate * p.grad

        norm_d = torch.norm(d).item()
        norm_d_mgda = torch.norm(d_mgda).item()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, d_MGDA = {norm_d_mgda}")
        track_loss1.append(loss1.item())
        track_loss2.append(loss2.item())
        track_d.append(norm_d)
        track_d_MGDA.append(norm_d_mgda)
        if norm_d_mgda < eps or norm_d < eps:
            print(f"EARLY STOPPING at epoch {epoch+1}/{num_epochs}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")
            break

    aggregator_path = os.path.join(path, aggregator.__name__)
    os.makedirs(aggregator_path, exist_ok=True) 

    model_path = os.path.join(aggregator_path, "model.pt")
    torch.save(model.state_dict(), model_path) # save model

    n_iterations = len(track_loss1) # save data
    iterations = np.arange(n_iterations)
    data = {
        "iterations" : iterations,
        "first loss function" : track_loss1,
        "second loss function" : track_loss2,
        "norm of d" : norm_d,
        "measure of Pareto stationarity" : norm_d_mgda
    }

    pickle_filename = f"{aggregator.__name__}_data.pkl"  # save as pickle file
    pickle_path = os.path.join(aggregator_path, pickle_filename) 
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    df = pd.DataFrame(data)
    csv_filename = f"{aggregator.__name__}_data.csv"
    csv_path = os.path.join(aggregator_path, csv_filename)
    df.to_csv(csv_path, index=False)

    return (track_loss1, track_loss2, track_d, track_d_MGDA)


if __name__ == "__main__":
    aggregator = UPGrad # will include other aggregator and different random seeds later

    track_loss1, track_loss2, track_d, track_d_MGDA = train(aggregator, MGDA)
    n_iterations = len(track_loss1)
    iterations = range(n_iterations)


    # plot and save each curve individually
    for title, curve in zip(["First loss function", "Second loss functions", "Norm of d" , "Measure of Pareto Stationarity"], [track_loss1, track_loss2, track_d, track_d_MGDA]):
        plt.plot(iterations, curve)
        plt.title(title)
        plt.xlabel("Number of iterations")
        plot_filename = f"{aggregator.__name__}_{title.replace(' ', '_')}.png"
        plot_path = os.path.join(path, aggregator.__name__, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # plot everything in one graph
    plt.subplot(2,2,1) 
    plt.plot(iterations, track_loss1)
    plt.title("First loss functions")
    plt.xlabel("Number of iterations")

    plt.subplot(2,2,2)
    plt.plot(iterations, track_loss2)
    plt.title("Second loss functions")
    plt.xlabel("Number of iterations")

    plt.subplot(2,2,3)
    plt.plot(iterations, track_d)
    plt.title("Norm of d")
    plt.xlabel("Number of iterations")

    plt.subplot(2,2,4)
    plt.plot(iterations, track_d_MGDA)
    plt.title("Measure of Pareto Stationarity")
    plt.xlabel("Number of iterations")

    plot_filename = f"{aggregator.__name__}_data.png"
    plot_path = os.path.join(path, aggregator.__name__, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #plt.show()


    success_rate, fairness = evaluate()
    print(f"Success rate = {success_rate}, Fairness = {fairness}")
    plot_filename = f"{aggregator.__name__}_evaluation.txt"
    txt_path = os.path.join(path, aggregator.__name__, plot_filename)
    np.savetxt(txt_path, [success_rate, fairness])

    
    



    
