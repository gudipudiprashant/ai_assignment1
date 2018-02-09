
from regression import Regression
from genetic_algo import GeneticAlgo
from simulated_annealing import SimulatedAnnealingAlgo, NumItersPerTempType

reg = Regression("train.csv")
reg.preprocess_data()
all_features = reg.get_all_features()
# print(reg.X_train.shape)
# print(reg.X_test.shape)
obj_fn = reg.get_accuracy
print("starting gentic algo")
# wrapper = GeneticAlgo(all_features, obj_fn)
# print(wrapper.run())

num_iters_obj = NumItersPerTempType("increasing", {'inc_iter': 200})
wrapper = SimulatedAnnealingAlgo(all_features, obj_fn,
                                start_temp=100, 
                                num_iter_per_temp_fn=num_iters_obj)
print(wrapper.run())

# print(wrapper.run())

# reg = Regression("train.csv")
# reg.preprocess_data()
# print(reg.X_train)
# print(reg.Y_train)
# print(reg.get_all_features())
# print(reg.get_log_MSE(reg.get_all_features()))
