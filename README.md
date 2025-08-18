# Dependancies 

You can install all required dependancies other than libmoon by doing `pip install -r requirements.txt`. The libmoon posted on PyPI does not contain all the required scirpts to run the fairness benchmark. It is recommended to install the library manually by cloning their repo: `git clone https://github.com/xzhang2523/libmoon`

# Files in this repo
1. run.py : run the experiment 
2. aggregators.py : all aggregators that would be used in the experiment
3. synthetic_problems.py : all synthetic problems that would be used in the experiment
4. fairness_classification_gpu.py : run the fairness benchmark 
5. fairness_classification_gpu.py : run the fairness benchmark with gpu acceleration
6. plot_from_data.py : plot graphs for synthetic problems
7. plot_fairness.py : plot graphs for two objective fairness benchmark
8. plot_fairness.py : plot graphs for three objective fairness benchmark

# How to run experiment 

To run experiment, run `python3 run_experiemtn.py`. When the code is running, the scipt would create folder `experiment result` to store all graphs, csv's, and pickle files. 

Similarly, 

To change experiment settings, edit these lists which can be found at the beginning of the code (or in the first line of `if __name__ == "__main__"` for fairness benchmark).

1. aggregators : Those aggregator that will be used in the experiment. To include more aggregtors, import the aggregators and include them in the array. Note that all aggregators are functions with typing `aggregators(jacobian : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`. The input is a $n$ by $m$ jacobian, where $n$ is the number of task, and $m$ is the dimension of $x$. The output is a tuple ($d$, $\alpha$). $d$ is a 1D tensor of length $m$ that represents the descent direction, and $\alpha$ 1D tensor of length $n$ that represents the weighting.

2. problems : The synthetic problem the experiment will be using. To include more problems, define the problems as a function `problem(x : torch.Tensor) -> List[torch.Tensor]`. The function will take in a 1D torch Tensor as input, and output a list of 1 by 1 tensor, each representing one function value of the objective function. Length of the output list = number of objective functions for the problem. Upon defining a new synthetic problem, the dimension of $x$ and the bounds of the problem should also be specified by changing the dictionary `problem_dim` and `bounds` respectively.

3. seeds : The random seeds that will be used in the experiment.

To change other parameters such as learning rate, tolarance, maximum number of iteration, etc., change their default value in `iteration()`(synthetic problems), or `train()` (fairness).

# Graph plotting

`run_experiemtn.py` will run the experiment and plot the corresponding graphs simultaneously. If only graph plotting is needed, one can store all pickle files in the folder `experiment result` folder under the correct subfolder. `experiment result` is organised as follows:

```bash
├── experiment result
│   ├── <problem_name_1>
│   │   ├── seed_<seed_number1>
├   │   ├── seed_<seed_number2>
│   │   │   ├──data
├   │   │   │   ├── <aggregator_name_1>*.pkl
├   │   │   │   ├── <aggregator_name_1>.pkl
├   │   │   │   ├── <aggregator_name_2>*.pkl
├   │   │   │   ├── <aggregator_name_2>.pkl
│   │   │   ├──plot
├   │   │   │   ├── combined.png
├   │   │   │   ├── combined2.png
├   │   │   │   ├── <aggregator_name_1>_normalised.png
├   │   │   │   ├── <aggregator_name_1>_unnormalised.png
├   │   │   │   ├── <aggregator_name_2>_normalised.png
├   │   │   │   ├── <aggregator_name_2>_unnormalised.png
│   ├── <problem_name_2>
│   ├── <problem_name_3>

```

Each pickle file should include the following columns:
- iteration : number of interations
- x_trajectory : each row is an array representing the value of x 
- y_trajectory : each row is an array representing the value of y
- norm_d : the norm of descent direction
- distance_PS : measure of Pareto stationarity, with is measured by the norm of the descent direction given by MGDA.
- alpha : each row is an array representing the value of alpha, the weighting vector.
- is_start_hitting : Optional. A list of Booleans. It indicates when does $x$ starts hitting the boundary.
- is_stop_hitting : Optional. A list of Booleans. It indicates when does $x$ stops hitting the boundary and re-renters the interior.

combined.png shows every plot contained in the file in one single image file. combined2.png does the same thing but plot those curve corresponding to normalised aggregators (no "\*" in the name) and those corresponding to unnormalised aggregators (with "\*" in the name) in different plot.


