
from regression import Regression
from genetic_algo import GeneticAlgo

reg = Regression("train.csv")
all_features = reg.get_all_features()
obj_fn = reg.get_MSE

wrapper = GeneticAlgo(all_features, obj_fn)
print(wrapper.run())