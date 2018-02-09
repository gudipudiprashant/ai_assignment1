
from regression import Regression
from genetic_algo import GeneticAlgo

reg = Regression("train.csv")
reg.preprocess_data()
all_features = reg.get_all_features()
obj_fn = reg.get_log_MSE

# wrapper = GeneticAlgo(all_features, obj_fn)
# print(wrapper.run())

# reg = Regression("train.csv")
# reg.preprocess_data()
# print(reg.X_train)
# print(reg.Y_train)
# print(reg.get_all_features())
# print(reg.get_log_MSE(reg.get_all_features()))
