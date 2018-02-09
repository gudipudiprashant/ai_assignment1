import bisect
import itertools
import random

from base_search_class import SearchAlgo

INF = float("Inf")
# Hyper-parameters
NUM_GENS = 10
K = 1.5
MUTATION = 1
INIT_POP_CONST = 10

debug = {"mut":0, "non":0}
class GeneticAlgo(SearchAlgo):
  """
  Class that implements Genetic Search Algorithm for Feature Selection.
  "Survival of the fittest":
  The Genetic Algorithm is a heuristic optimization method
  inspired by the principles of natural evolution.
  """
  def __init__(self, all_features, obj_fn, num_gens=NUM_GENS, k=K,
    mutation_const=MUTATION, pop_const=INIT_POP_CONST):
    """
    Constructor to create an object of the class.
    Args:
      all_features  : Argument to base class - SearchAlgo
      obj_fn        : Argument to base class - SearchAlgo
      num_gens      : (int) Parameter to control the number of runs/generations
                      of the search algorithm.
      k             : (int) Hyper-parameter used for Rank-based fitness. Value
                      should be between 1 and 2
      mutation_const: (float) mutation rate constant. The actual mutation rate
                      will be 1/(mutation_const * length_chromosome)
      pop_const     : (int) specifies the size of population where
                      population size = pop_const * num_features.
                      ***Preferably even number***
    """
    self.num_gens = num_gens
    self.k = k
    self.mutation_const = mutation_const
    self.pop_const = pop_const
    super(GeneticAlgo, self).__init__(all_features, obj_fn)

  def run(self):
    """
    Method to be called to run the search algorithm. It creates an initial
    population and then creates subsequent generations based on the GA
    operators like selection, cross-over and mutation. For each generation,
    it updates the best solution seen and outputs the best solution after
    all the generations.

    Returns:
      Python list of feature names, corresponding to the best state/
      solution found by the GA search algorithm.
    """
    cur_best_chrom = ""
    cur_best_score = -INF
    # Initial population
    population = self.initialization()
    generation = 0
    while True:
      generation += 1
      # Get the scores of each chromosome based on the objective function
      scores = [(self.getScore(chrom), chrom) for chrom in population]

      # Update the best chromosome seen
      for score, chromosome in scores:
        if score > cur_best_score:
          cur_best_chrom = chromosome
          cur_best_score = score

      # Stop the algorithm if num of generations have exceeded num_gens set
      if generation >= self.num_gens:
        break

      # Calculate Rank-based fitness values
      fitness_val_list = self.get_fitness_val_population(scores)

      # Select chromosomes for cross-over
      population_sz = len(population)
      # We select n chromosomes for cross-over if there are n chromosomes
      # in population and create 2 offsprings for each pair to maintain constant
      # population size
      selection_sz = population_sz
      # The chromosomes are selected based on Roulette-Wheel stochastic sampling
      # with replacement
      selected_population = self.roulette_wheel_selection(
        fitness_val_list, selection_sz)
      
      # Performing cross-over with mutation to generate new chromosomes
      new_population = []
      # Pair the chromosomes for cross-over
      for i in range(1, selection_sz, 2):
        new_population.extend(self.uniform_crossover(
          selected_population[i-1], selected_population[i]))

      # Continue the above steps with the generated new population
      population = new_population

    # Return the best seen solution/chromosome
    return self.decodeFeatures(cur_best_chrom)

  def get_random_chromosome(self, chrom_length):
    """
    Generates a random chromosome/binary string of size chrom_length
    Args:
      chrom_length  : length of the random chromosome to be generated
    Returns:
      (str) Binary String of length chrom_length with bits chosen uniformly
      at random. 
    """
    chrom = ""
    for i in range(chrom_length):
      # get a random bit
      chrom += str(random.randrange(2))
    return chrom

  def get_fitness_val_population(self, scores):
    """
    Returns a list of 2-tup (chromosome, fitness_val) where fitness_val
    represents how fit the chromosome is within the population.
    Args:
      scores  : List of 2-tup (score, chromosome)
    Returns:
      List of 2-tup (chromosome(str), fitness_val(float))
    """
    # Calculating "Rank-based" fitness values
    # Sort scores
    scores = sorted(scores)

    fitness_val_list = []
    # Fitness val is calculated as index(1-based) * k
    # after sorting the scores, i.e, high score gets high fitness.
    for i in range(len(scores)):
      fitness_val_list.append((scores[i][1], (i+1)*self.k))
    return fitness_val_list

  def mutate(self, chromosome):
    """
    Given a chromosome, mutates/flips a random gene with probability
    mutation_rate.
    Args:
      chromosome    : (str) binary string
    Returns:
      A mutated chromosome(str)
    """
    mutation_rate = 1/(len(chromosome) * self.mutation_const)
    mutated = ""
    for allele in chromosome:
      if random.random() < mutation_rate:
        mutated += str((int(allele) + 1)%2)
      else:
        mutated += allele

    return mutated

  def uniform_crossover(self, p1, p2, num_offspring=2):
    """
    Applies uniform cross-over operator on the parent chromosomes to
    generate new off-spring chromomsomes.
    Args:
      p1            : (str) chromosome/binary string
      p2            : (str) same as p1
      num_offspring : (int) number of offspring chromosomes to be generated
    Returns:
      Python list of (str) offspring chromosomes.
    """
    chrom_length = len(p1)
    offsprings = ["" for i in range(num_offspring)]
    
    for i in range(num_offspring):
      for gene in range(chrom_length):
        # With 0.5 probability use p1's gene, else p2's gene
        if random.randrange(2):
          offsprings[i] += p1[gene]
        else:
          offsprings[i] += p2[gene]
        # The mutation rate is set as 1/(mutation_const * Num of features)
      offsprings[i] = self.mutate(offsprings[i])
    return offsprings

  def roulette_wheel_selection(self, fitness_val_list, sel_sz):
    """
    Returns sel_sz number of chromosomes for cross-over based on
    Roulette Wheel stochastic selection with replacement.
    """
    selected_pop = []
    # unpack the fitness values and chromosomes
    choices, weights = zip(*fitness_val_list)
    # Calculate the cumulative sum of fitness values
    cumdist = list(itertools.accumulate(weights))
    # choose sel_sz number of choices with probability proportional to their
    # weights
    for i in range(sel_sz):
      x = random.random()*cumdist[-1]
      # Check which bucket/bin the random number falls into
      selected_pop.append(choices[bisect.bisect(cumdist, x)])
    return selected_pop

  def initialization(self):
    """
    Initializes the population of size (num_features * pop_const)
    as random chromosomes.
    Returns:
      List of (str) chromosomes generated uniformly at random.
    """
    chrom_length = len(self.all_features)
    # Initial population contains 10*N chromosomes/strings
    return [self.get_random_chromosome(chrom_length) 
        for i in range(self.pop_const * chrom_length)]


