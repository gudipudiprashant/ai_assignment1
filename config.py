# Config file to take input parameters and hyper-parameters

# Input Training file(only csv)
train_file = "train_house_price.csv"

# Is the problem regression or classsification?
is_regr = True

# Use Genetic Search or Simulated Annealing?
is_gen = False

# Find the optimal solution using brute force search
# Warning: Only recommended for small search space
find_optim = False

# Print graph of obj fn with each step
print_graph = True


# GENETIC ALGO : HYPER-PARAMETERS
# Number of generations
NUM_GENS = 30
# k for Rank-based fitness calculation - should be between 1 and 2
K = 1.5
# (Float)Mutation rate constant. The actual mutation rate
# will be 1/(mutation_const * length_chromosome)
MUTATION = 1
# Population constant
# (Expected even number, else population changes between first and second gen)
# population size = pop_const * num_features
POP_CONST = 10


# SIMULATED ANNEALING : HYPER-PARAMETERS

# Cooling function parameters

#Options: linear, geometric, slow_decrease
COOLING_TYPE = "slow_decrease"
#Options: lin_cooling, geom_cooling, slow_cooling
COOLING_FACTORS = {'slow_cooling': 10}

# Number of iterations per temperature parameters

#Options: constant, increasing
NUM_ITERS_TYPE = "constant"
#Options: const_iter, inc_iter
NUM_ITERS_FACTORS = {'constant': 1}

# Simulated annealing parameters
START_TEMP = 2000
FINAL_TEMP = 0

#Constant to tune the probability of accepting a worse solution.
ACCEPT_PROBABILITY_FACTOR = 0.000001