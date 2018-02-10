import random
import math

from base_search_class import SearchAlgo
import config


# Temperature decrease hyper-parameters
DEFAULT_LIN_COOLING = 1 
DEFAULT_GEOM_COOLING = 0.9
DEFAULT_SLOW_COOLING = 2

#Number of Iterations hyper-parameters
DEFAULT_CONST_ITERS = 1
DEFAULT_INC_ITERS = 100

#Simulated annealing hyper-parameters
ACCEPTANCE_PARAMETER = config.ACCEPT_PROBABILITY_FACTOR

class CoolingType:
    """
    Describes how temperature decreases with the number of steps
    completed in the algorithm.
    The new temperature depends on the current temperature.
    """

    def __init__(self, type_name="linear", type_factor_dict={}):
        """
        Initialize the object

        Args:
        type_name       :  Type of function chosen.
        type_factor_dict:  Hyper-parameters for the function
        """

        # Shows the types of functions available
        self.dict_types = { 
                           "linear": self.cooling_linear,
                           "geometric": self.cooling_geometric,
                           "slow_decrease": self.cooling_slow_decr
                          }
        
        self.type_fn = self.dict_types[type_name]
        self.type_factor_dict = type_factor_dict
        
    def get_new_temp(self, cur_temp):
        """
        Calculates new temperature based on current temperature using the
        function type and parameters chosen while creating the object.

        Args:
        cur_temp: Current Temperature

        Returns new temperature.
        """

        return self.type_fn(cur_temp)

    ## Private Functions

    def cooling_linear(self, cur_temp):
        """
        Linear temperature decrease.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        lin_cooling = self.type_factor_dict.get('lin_cooling',
                                                  DEFAULT_LIN_COOLING)
        return cur_temp - lin_cooling

    def cooling_geometric(self, cur_temp):
        """
        Geometric temperature decrease.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        geom_cooling = self.type_factor_dict.get('geom_cooling',
                                                    DEFAULT_GEOM_COOLING)
        return cur_temp*geom_cooling

    def cooling_slow_decr(self, cur_temp):
        """
        Slow temperature decrease. Usually paired with 1 iteration
        per temperature. The motivation is to keep the number of
        iterations per temperature low, but decrease the temperature
        slowly to allow exploration of the state space.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        slow_cooling = self.type_factor_dict.get('slow_cooling', DEFAULT_SLOW_COOLING)
        return float(cur_temp) / (1 + (float(1)/(slow_cooling*cur_temp)))

class NumItersPerTempType:
    """
    Describes how the number of iterations per temperature varies
    with the number of steps completed in the algorithm.
    The number of iterations depends on the current temperature.
    """

    def __init__(self, type_name="constant", type_factor_dict={}):
        """
        Initialize the object

        Args:
        type_name       :  Type of function chosen.
        type_factor_dict:  Hyper-parameters for the function.
        """

        # Shows the types of functions available.
        self.dict_types = {
                            "constant": self.num_iters_constant,
                            "increasing": self.num_iters_inc
                          }

        self.type_fn = self.dict_types[type_name]
        self.type_factor_dict = type_factor_dict


    def get_num_iters(self, cur_temp):
        """
        Returns number of iterations to be performed per temperature value.

        Args:
        cur_temp: Current Temperature

        Returns number of iterations at the temperature.
        """

        return self.type_fn(cur_temp)

    ##Private Functions

    def num_iters_constant(self, cur_temp):
        """
        Constant number of iterations at any temperature.

        Args:
        cur_temp: Current Temperature

        Returns number of iterations at the temperature.
        """

        const_iter = self.type_factor_dict.get('const_iter',
                                               DEFAULT_CONST_ITERS)
        return const_iter

    def num_iters_inc(self, cur_temp):
        """
        Increasing number of iterations at any temperature.
        No. of iterations = (Proportionality constant) / (current temperature)

        Args:
        cur_temp: Current Temperature

        Returns number of iterations at the temperature.
        """

        inc_iter_factor = self.type_factor_dict.get('inc_iter', 
                                                    DEFAULT_INC_ITERS)
        return int(inc_iter_factor/cur_temp)


class SimulatedAnnealingAlgo(SearchAlgo):
    """
    Executes the Simulated Annealing algorithm.
    """

    def __init__(self, all_features, obj_fn, start_temp=100, final_temp=0,
                 num_iter_per_temp_fn=NumItersPerTempType(),
                 cooling_fn=CoolingType()):
        """
        Initialize the algorithm object.

        Args:
        all_features        :  Argument to the base class - SearchAlgo
        obj_fn              :  Argument to the base class - SearchAlgo
        start_temp          :  Starting temperature
        final_temp          :  Stopping temperature
        num_iter_per_temp_fn:  NumItersPerTempType object
        cooling_fn          :  CoolingType object
        """

        super(SimulatedAnnealingAlgo, self).__init__(all_features, obj_fn)

        # Temperature parameters
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.cooling_fn = cooling_fn
        self.num_iter_per_temp_fn = num_iter_per_temp_fn

        # Variables for the algorithm
        self.cur_temp = None
        self.cur_state = None #encoded subset of features.
        self.cur_state_energy = None

    def run(self):
        """
        Executes the simulated annealing algorithm.

        Returns the decoded final state (subset of features) and its score.
        """

        self.initialize()

        # Loop executes till final_temp reached.
        self.energy_list = []
        self.temp_list = []
        self.best_energy_temp = -10
        self.best_energy = -1000
        self.best_energy_state = None

        # counter for printing
        counter = 0
        while(True):
            # Print every 1000 temperature iterations
            counter += 1
            if counter%1000 == 0:
                print("At temperature ", self.cur_temp,
                      ", Current Energy: ", self.cur_state_energy,
                      " Best Energy: ", self.best_energy)

            if self.cur_temp <= self.final_temp:
                break

            # Iterations per temperature
            for cur_iter in range(self.num_iter_per_temp_fn.get_num_iters(self.cur_temp)):

                # Generate random neighbour
                neighbour_state = self.create_random_neighbour()
                neighbour_state_energy = self.getScore(neighbour_state)

                if(neighbour_state_energy > self.cur_state_energy):
                    # Jump to neighbouring state if it has more energy.
                    self.cur_state = neighbour_state
                    self.cur_state_energy = neighbour_state_energy
                else:
                    # The neighbouring state is worse-off.
                    energy_diff = neighbour_state_energy - self.cur_state_energy
                    
                    # Jump to neighbouring state with probability.
                    if random.random() < math.exp(float(energy_diff)/(ACCEPTANCE_PARAMETER*self.cur_temp)):
                        self.cur_state = neighbour_state
                        self.cur_state_energy = neighbour_state_energy

                #Get the best energy state
                if self.cur_state_energy > self.best_energy:
                    self.best_energy_temp = self.cur_temp
                    self.best_energy = self.cur_state_energy
                    self.best_energy_state = self.cur_state

            if neighbour_state_energy > -0.1:
                self.energy_list.append(self.cur_state_energy)
                self.temp_list.append(self.cur_temp)

            self.cur_temp = self.cooling_fn.get_new_temp(self.cur_temp)

        print("Ans: ",self.cur_state_energy)
        print("Best: ", self.best_energy, self.best_energy_temp)

        return self.decodeFeatures(self.cur_state), self.cur_state_energy

    def get_graph(self):
        """
        Prints the graph
        """
        import matplotlib.pyplot as plt
        import numpy as np
        x = self.temp_list
        y = self.energy_list
        plt.plot(x,y)
        plt.xlim(self.start_temp, 0)
        plt.xlabel("Temperature")
        plt.ylabel("Final energy value of state")
        plt.title("Simulated Annealing Algo")
        plt.legend()
        plt.show()

    #Private Functions

    def initialize(self):
        """
        Initializes the variables of the algorithm.
        """

        self.cur_temp = self.start_temp
        self.cur_state = self.create_random_initial_state()
        self.cur_state_energy = self.getScore(self.cur_state)


    def create_random_initial_state(self):
        """
        Creates random state (random subset of features).

        Returns the generated random state.
        """

        state = []

        for elem in self.all_features:
            if random.choice([True, False]):
                state.append(elem)

        return self.encodeFeatures(state)

    def create_random_neighbour(self):
        """
        Creates random neighbour state using the current state.

        Returns the generated neighbouring state.
        """
        
        current_state = self.cur_state

        #randint includes both endpoints.
        feature_to_flip = random.randint(0,len(current_state)-1)
        
        #generating the neighbour.
        neighbour = ''
        for i in range(len(current_state)):
            if i == feature_to_flip:
                neighbour += str((int(current_state[feature_to_flip]) + 1) % 2)
            else:                
                neighbour += current_state[i]

        return neighbour

