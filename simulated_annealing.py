import random
import math

from base_search_class import SearchAlgo


# Temperature decrease hyper-parameters
default_lin_temp_decr = 1 
default_geom_temp_decr = 0.9
default_slow_decr_temp_decr = 2

#Number of Iterations hyper-parameters
default_const_iters = 1
default_inc_iters = 100

class TemperatureDecrType:
    self.dict_types = { 
                        "linear": self.temp_decr_linear,
                        "geometric": self.temp_decr_geometric,
                        "slow_decrease": self.temp_decr_slow_decr
                      }

    self.type_fn = None
    self.type_factor_dict = None


    def __init__(self, type_name="linear", type_factor_dict={}):
        self.type_fn = self.dict_types[type_name]
        self.type_factor_dict = type_factor_dict

    def get_new_temp(self, cur_temp):
        return self.type_fn(cur_temp)

    # Private Functions

    def temp_decr_linear(self, cur_temp):

        lin_temp_decr = self.type_factor_dict.get('lin_decr', default_lin_temp_decr)
        return cur_temp - lin_temp_decr

    def temp_decr_geometric(self, cur_temp):

        geom_temp_decr = self.type_factor_dict.get('geom_decr', default_geom_temp_decr)
        return cur_temp*geom_temp_decr

    def temp_decr_slow_decr(self, cur_temp):

        slow_decr_temp_factor = self.type_factor_dict.get('slow_decr', default_slow_decr_temp_decr)
        return float(cur_temp) / (1 + (float(1)/(slow_decr_temp_factor*cur_temp)))

class NumItersPerTempType:

    self.dict_types = {
                        "constant": self.num_iters_constant,
                        "increasing": self.num_iters_inc
                      }

    self.type_fn = None
    self.type_factor_dict = None

    def __init__(self, type_name="constant", type_factor_dict={}):
        self.type_fn = self.dict_types[type_name]
        self.type_factor_dict = type_factor_dict

    def get_num_iters(self, cur_temp):
        return self.type_fn(cur_temp)

    #Private Functions

    def num_iters_constant(self, cur_temp):

        const_iter = self.type_factor_dict.get('const_iter', default_const_iters)
        return const_iter

    def num_iters_inc(self, cur_temp):

        inc_iter_factor = self.type_factor_dict.get('inc_iter', default_inc_iters)
        return float(inc_num_iters_factor)/cur_temp


class SimulatedAnnealingAlgo(SearchAlgo):

    # Temperature parameters
    self.start_temp = None
    self.final_temp = None
    self.num_iter_per_temp_fn = None
    self.temp_decr_fn = None


    # State variables of algorithm
    self.cur_temp = None
    self.cur_state = None
    self.cur_state_score = None

    def __init__(self,
                 all_features,
                 obj_fn,
                 start_temp=100,
                 final_temp=0,
                 num_iter_per_temp_fn=NumItersPerTempType(),
                 temp_decr_fn=TemperatureDecrType()):

        super(SimulatedAnnealing, self).__init__(all_features, obj_fn)

        self.start_temp = start_temp
        self.final_temp = final_temp
        self.temp_decr_fn = temp_decr_fn
        self.num_iter_per_temp_fn = num_iter_per_temp_fn


    def run(self):

        self.initialize()

        while(True):
            if self.cur_temp <= self.final_temp:
                break

            for cur_iter in range(self.num_iter_per_temp_fn.get_num_iters(self.cur_temp)):

                neighbour_state = self.create_random_neighbour()
                neighbour_state_score = self.getScore(neighbour_state)

                if(neighbour_state_score > self.cur_state_score):
                    self.cur_state = neighbour_state
                    self.cur_state_score = neighbour_state_score
                else:
                    performance_diff = neighbour_state_score - self.cur_state_score

                    if random.random() < math.exp(float(performance_diff)/self.cur_temp):
                        self.cur_state = neighbour_state
                        self.cur_state_score = neighbour_state_score

            self.cur_temp = self.temp_decr_fn.get_new_temp(self.cur_temp)

        return self.cur_state


    #Private Functions

    def initialize(self):

        self.cur_temp = self.start_temp
        self.cur_state = self.create_random_initial_state()
        self.cur_state_score = self.getScore(self.cur_state)


    def create_random_initial_state(self):

        state = []

        for elem in self.all_features:
            if random.choice([True, False]):
                state.append(elem)

        return self.encodeFeatures(state)

    def create_random_neighbour(self):

        current_state = self.cur_state

        feature_to_flip = random.randint(0,len(current_state))
        neighbour = current_state
        neighbour[feature_to_flip] = str((int(current_state[feature_to_flip]) + 1) % 2)

        return neighbour

