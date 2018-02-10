## Feature Selection using Wrapper Method
The python implementation of feature selection using Genetic Algorithm and Simulated annealing

### Dependencies
1. SciKit-learn
2. Numpy
3. Matplotlib

### Clone and Run
```bash
>>> git clone https://github.com/gudipudiprashant/ai_assignment1
>>> cd ai_assignment1
>>> python3 test.py
```

To run the script with necessary parameters (dataset, printing graph, genetic/simulate annealing), you need to change the variables in `config.py`. It contains input parameters and the hyperparameters for both the algorithms.

The following input parameters can be set in `config.py`

1. `train_file`: The file containing the dataset to train on
2. `is_regr`: `True` if the problem is regression and `False` for classification
3. `is_gen`: `True` for Genetic algorithm and `False` for simulated annealing
4. `find_optim`: `True` for using exastive search to find the globally optimal solution
5. `print_graph`: `True` to show graph

**Note:** The hyperparameters are tuned for the best performace with the current dataset `train_house_price.csv` and may not perform optimally on other datasets.
