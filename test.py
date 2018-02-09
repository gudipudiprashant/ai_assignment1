
from regression import Regression
from genetic_algo import GeneticAlgo
from simulated_annealing import SimulatedAnnealingAlgo, NumItersPerTempType

reg = Regression("train.csv")
reg.preprocess_data()
all_features = reg.get_all_features()
print("Training data: " ,reg.X_train.shape)
print("Testing data: ", reg.X_test.shape)
obj_fn = reg.get_accuracy
print("starting Simulated Annealing algorithm:")

num_iters_obj = NumItersPerTempType("increasing", {'inc_iter': 200})
wrapper = SimulatedAnnealingAlgo(all_features, obj_fn,
                                start_temp=100, 
                                num_iter_per_temp_fn=num_iters_obj)
print(wrapper.run())

# wrapper = GeneticAlgo(all_features, obj_fn)
# print(wrapper.run())

