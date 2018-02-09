
from regression import Regression
from genetic_algo import GeneticAlgo, debug
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
print("starting gentic algo")
wrapper = GeneticAlgo(all_features, obj_fn)
print(wrapper.run())
# print(debug)
# def f1(x,y,z):
#     return x+y+z

# x=2
# y=3
# def f2(z):
#     return lambda z: f1(x,y,z) 

# print(f2(5))
# num_iters_obj = NumItersPerTempType("increasing", {'inc_iter': 200})
# wrapper = SimulatedAnnealingAlgo(all_features, obj_fn,
#                                 start_temp=100, 
#                                 num_iter_per_temp_fn=num_iters_obj)
# print(wrapper.run())

# wrapper = GeneticAlgo(all_features, obj_fn)
# print(wrapper.run())

