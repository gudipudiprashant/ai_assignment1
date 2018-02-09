# Code for testing
from genetic_algo import GeneticAlgo
from regression import Supervisor
from simulated_annealing import SimulatedAnnealingAlgo, NumItersPerTempType


def main():
    train_file = "train_house_price.csv" 
    # input("Input Training file(only csv): ")
    is_regr = input("Is it a regression problem (Y/N): ")
    if "y" in is_regr.lower():
        is_regr = True
    else:
        is_regr = False
    # Create Supervisor object for testing the feature subset
    model = Supervisor(train_file, is_regr)
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

    print("Starting Genetic algo")
    wrapper = SimulatedAnnealingAlgo(all_features, obj_fn) 
    # GeneticAlgo(all_features, obj_fn)
    # Runs the search algorithm and returns the best solution found
    feature_subset = wrapper.run()[0]
    print("Feature subset: ", feature_subset)
    print("Score: ", obj_fn(tuple(feature_subset)))
    print(model.get_obj_fn.cache_info())
# Code to get the optimal feature subset by search the entire solution space
# max_score = -1
# best_str = ""
# for i in range(0, 2**12):
#     enc_str = "{0:011b}".format(i)
#     sc = wrapper.getScore(enc_str)
#     if sc > max_score:
#         max_score = sc
#         best_str = enc_str
# print(best_str, max_score)

# wrapper = SimulatedAnnealingAlgo(all_features, obj_fn,
#                                 start_temp=100, 
#                                 num_iter_per_temp_fn=num_iters_obj)
# print(wrapper.run())

# wrapper = GeneticAlgo(all_features, obj_fn)
# print(wrapper.run())

if __name__ == "__main__":
    main()
