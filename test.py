# Code for testing

import config 

from genetic_algo import GeneticAlgo
from regression import Supervisor
from simulated_annealing import SimulatedAnnealingAlgo, NumItersPerTempType


def main():
    # Create Supervisor object for testing the feature subset
    model = Supervisor(config.train_file, config.is_regr)
    # pre-process/clean the data
    model.preprocess_data()
    # get the list of features
    all_features = model.get_all_features()
    print("Training data: " ,model.X_train.shape)
    print("Testing data: ", model.X_test.shape)
    # The objective fn which is used a black box by the wrapper method
    # to check the performance of feature subset
    obj_fn = model.get_obj_fn

    # The wrapper method using the Search Algorithm to find the optimal feature
    # subset - feature selection
    if config.is_gen:
        print("Starting Genetic algo")
        wrapper = GeneticAlgo(all_features, obj_fn)
    else:
        print("Starting Simulated Annealing")
        wrapper = SimulatedAnnealingAlgo(all_features, obj_fn) 

    # Runs the search algorithm and returns the best solution found
    feature_subset = wrapper.run()[0]
    # printing the best found feature subset
    print("\nFeature subset: ", feature_subset)
    print("\n Length of feature subset found: ", len(feature_subset),
        "Length of all features: ", len(wrapper.all_features))
    # Printing the score/obj fn value of using all the features 
    print("\nScore using all the features: ",
        obj_fn(tuple(wrapper.all_features)))
    #Printing the obj_fn value of the found feature subset
    print("\nScore of the best feature subset found: ", 
        obj_fn(tuple(feature_subset)))
    # Printing the cache results - which gives an idea of how many states
    # were explored by the search algorithm
    print("\n",model.get_obj_fn.cache_info())

    if config.find_optim:
        find_optimal(wrapper)
    if config.print_graph:
        wrapper.get_graph()

# Code to get the optimal feature subset by search the entire solution space
def find_optimal(wrapper):
    max_score = -10**8
    best_str = ""
    str_sz = len(wrapper.all_features)
    for i in range(0, 2**(str_sz+1)):
        bin_format = "{0:0"+str(str_sz)+"b}" 
        enc_str = bin_format.format(i)
        sc = wrapper.getScore(enc_str)
        if sc > max_score:
            max_score = sc
            best_str = enc_str
    print("\nMaximum possible score: ", max_score)
    print("Corresponding features: ", wrapper.decodeFeatures(best_str))

if __name__ == "__main__":
    main()
