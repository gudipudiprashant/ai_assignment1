import random
import math

from base_search_class import SearchAlgo


# Temperature decrease hyper-parameters
DEFAULT_LIN_TEMP_DECR = 1 
DEFAULT_GEOM_TEMP_DECR = 0.9
DEFAULT_SLOW_DECR_TEMP_DECR = 2

#Number of Iterations hyper-parameters
DEFAULT_CONST_ITERS = 1
DEFAULT_INC_ITERS = 100

class TemperatureDecrType:
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
                           "linear": self.temp_decr_linear,
                           "geometric": self.temp_decr_geometric,
                           "slow_decrease": self.temp_decr_slow_decr
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

    def temp_decr_linear(self, cur_temp):
        """
        Linear temperature decrease.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        lin_temp_decr = self.type_factor_dict.get('lin_decr',
                                                  DEFAULT_LIN_TEMP_DECR)
        return cur_temp - lin_temp_decr

    def temp_decr_geometric(self, cur_temp):
        """
        Geometric temperature decrease.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        geom_temp_decr = self.type_factor_dict.get('geom_decr',
                                                    DEFAULT_GEOM_TEMP_DECR)
        return cur_temp*geom_temp_decr

    def temp_decr_slow_decr(self, cur_temp):
        """
        Slow temperature decrease. Usually paired with 1 iteration
        per temperature. The motivation is to keep the number of
        iterations per temperature low, but decrease the temperature
        slowly to allow exploration of the state space.

        Args:
        cur_temp: Current Temperature

        Returns new temperature
        """

        slow_decr_temp_factor = self.type_factor_dict.get('slow_decr', DEFAULT_SLOW_DECR_TEMP_DECR)
        return float(cur_temp) / (1 + (float(1)/(slow_decr_temp_factor*cur_temp)))

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
                 temp_decr_fn=TemperatureDecrType()):
        """
        Initialize the algorithm object.

        Args:
        all_features        :  Argument to the base class - SearchAlgo
        obj_fn              :  Argument to the base class - SearchAlgo
        start_temp          :  Starting temperature
        final_temp          :  Stopping temperature
        num_iter_per_temp_fn:  NumItersPerTempType object
        temp_decr_fn        :  TemperatureDecrType object
        """

        super(SimulatedAnnealingAlgo, self).__init__(all_features, obj_fn)

        # Temperature parameters
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.temp_decr_fn = temp_decr_fn
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
        while(True):

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
                    if random.random() < math.exp(float(energy_diff)/self.cur_temp):
                        self.cur_state = neighbour_state
                        self.cur_state_energy = neighbour_state_energy

            self.cur_temp = self.temp_decr_fn.get_new_temp(self.cur_temp)

        return self.decodeFeatures(self.cur_state), self.cur_state_energy


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

