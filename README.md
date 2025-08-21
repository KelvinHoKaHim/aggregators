
## Dependencies 

You can install all required dependencies other than libmoon through pip: 
```
pip install -r requirements.txt
``` 
We have made custom modification to libmoon, including introducing custom loss functions and commenting out import statements irrelevant to our experiments. We have provided the modified libmoon library in this repository. The codebase for the original libmoon library can be found here : https://github.com/xzhang2523/libmoon

## Running the experiments

- To run the synthetic benchmark: 
```
python3 -m synthetic_benchmark.run
```
- To run the two objective fairness benchmark: 
```
python3 -m fairness_benchmark_2.run
```
- To run the three objective fairness benchmark: 
```
python3 -m fairness_benchmark_3.run
```

After running the scripts, the experiment results can be found in `synthetic_benchmark/results`, `fairness_benchmark_2/results`, and `fairness_benchmark_3/results`. 

### Options

These are the default settings for the three experiments:

|                    | Synthetic            | Fairness (2 objectives) | Fairness (3 objectives) |
|--------------------|----------------------|-------------------------|-------------------------|
| Aggregators        | All^[1]              | All^[1]                 | All^[1]                |
| Synthetic problems | VLMOP2 & Omnitest    | N/A                     | N/A                     |
| Seeds              | 24, 42, 48, 100, 123 | 24, 42, 64, 100         | 24, 42, 64, 100         |
| Learning rate      | 0.001                | 0.005                   | 0.005                   |
| Tolerance          | 1e-2                 | 1e-3                    | 1e-3                    |


[1]: Refers to MGDA, Nash-MTL, Nash-MTL\*, UPGrad, UPGrad\*, DualProj, and DualProj\*

The settings can be changed by modifying the hyperparameter below:
- `--aggregator` : This hyperparameter takes in one or more values. By specifying the names of the aggregators, `run.py` would only run the specified aggregators instead of all of them.

- `--problems` : **(Only for synthetic benchmark)** Similar to `--aggregator`, by specifying the names of the synthetic problems, `run.py` would only run the specified problems instead of all of them.

- `--seeds` : Specifying one or more random seeds for the experiment

- `--lr` : Learning rate

- `--epochs` : Maximum number of epochs

- `--eps` : Stopping criteria tolerance 

## Graph plotting 
After running the experiments, you may plot the corresponding graphs by running the following commands

- Synthetic benchmark: 
```
python3 -m synthetic_benchmark.plot
```
- Two objective fairness benchmark: 
```
python3 -m fairness_benchmark_2.plot
```
- Three objective fairness benchmark: 
```
python3 -m fairness_benchmark_3.plot
```

## Options

The following options are provided for more flexibility
- `--only_plot_aggregators` : If specified, the script will only plot graphs with the specified aggregators. If not specified, the script will plot for every aggregator contained within `results`

- `--only_plot_seeds` : If specified, the script will only plot graphs with the specified seeds. If not specified, the script will plot for every seed contained within `results`

- `--format` : Format of image files. Set the `"png"` by default

After running the scripts, the plots can also be founds in `synthetic_benchmark/results`, `fairness_benchmark_2/results`, and `fairness_benchmark_3/results`. 

In each plot folder, you will find the following plots

1. Synthetic benchmarks
- First_loss_function.png : The loss curve of the first loss function
- Second_loss_function.png : The loss curve of the second loss function
- d.png : The norm of the descent direction ($\|d\|$) in each epoch
- Measure_of_Pareto_stationarity.png : The measure of Pareto stationarity in each epoch
- Combined.png : Every plots above being combined into one single image file
- Combined2.png : Similar to Combined.png, but instead of plotting every relevant curves in one single subplot, we plot them in two.

2. Two objectives fairness benchmark
- Cross_entropy_loss.png : The loss curve of the cross-entropy loss
- DEO.png : The loss curve of the DEO loss
- d.png : Same as above
- Measure_of_Pareto_stationarity.png : Same as above
- Combined.png : Same as above
- Combined2.png : Same as above

3. Three objectives fairness benchmark
- Cross_entropy_loss.png : Same as above
- DEO1.png : The loss curve of the DEO1^[2] loss
- DEO1.png : The loss curve of the DEO2 loss
- Measure_of_Pareto_stationarity.png : Same as above
- Combined.png : Same as above
- Combined2.png : Same as above

[2]: DEO1 is the same as DEO. We call it DEO1 in order to differentiate it from DEO2. 