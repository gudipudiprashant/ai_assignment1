import bisect
import itertools
import random

from base_search_class import SearchAlgo


class GeneticAlgo(SearchAlgo):
  def __init__(self, all_features, obj_fn, num_steps=100):
    self.num_steps = num_steps
    super(A, self).__init__(all_features, obj_fn)

  def run(self):
    cur_best_chr = ""
    cur_best_score = -10**6
    # Think about the termination condition
    population = self.initialization()
    step = 0
    while True:
      step += 1
      scores = [(self.getScore(x), x) for x in population]
      # Check for new best chromosome
      for score, chromosome in scores:
        if score > cur_best_score:
          cur_best_chr = chromosome
      # Stop the algorithm if num of steps have exceeded the set limit
      if step >= self.num_steps:
        break
      # Calculate rank-based fitness values
      fitness_val_list = self.get_fitness_val_population(scores)
      # Select chromosomes for cross-over
      n = len(fitness_val_list)
      # Hyper-parameters 
      ############# RE-READ THIS PART ###############
      selection_sz = n
      # num_offsprings = (n//selection_sz)*2
      # Select selection_sz chromosomes where n - size of population
      selected_population = self.roulette_wheel_selection(
        fitness_val_list, selection_sz)
      # Do cross-over
      new_population = []
      for i in range(0, selection_sz, 2):
        new_population.append(self.uniform_crossover(
          selected_population[i], selected_population[i+1]))

      population = new_population

    return self.decodeFeatures(cur_best_chr)

  def get_random_chromosome(length):
    chrom = ""
    for i in range(length):
      # get a random bit
      chrom += str(random.randrange(2))
    return chrom

  def get_fitness_val_population(self, scores, k=1.5):
    """
    Returns a list of 2-tup (chromosome, fitness_val)
    """
    # Calculating "Rank-based" fitness values
    scores = sorted(scores)
    
    fitness_val_list = []
    # Fitness val is calculated as index * k
    for i in range(m):
      fitness_val_list.append((scores[i][1], (i+1)*k))
    return fitness_val_list

  def mutate(self, allele, mutation_rate):
    if random.random() < mutation_rate:
      return str((int(allele) + 1)%2)
    else:
      return allele

  def uniform_crossover(self, p1, p2, num_offspring=2):
    length = len(p1)
    offsprings = ["" for i in range(num_offspring)]
    for i in range(num_offspring):
      for j in range(length):
        # With 0.5 probability use p1's gene, else p2's gene
        allele = ""
        if random.randrange(2):
          allele = p1[j]
        else:
          allele = p2[j]
        # The mutation rate is set as 1/(Num of features)
        offsprings[i] += self.mutate(allele, 1/length)
    return offsprings

  def roulette_wheel_selection(self, fitness_val_list, sel_sz):
    """
    Returns sel_sz number of chromosomes for cross-over based on
    Roulette Wheel stochastic selection.
    """
    selected_pop = []
    choices, weights = zip(*weighted_choices)
    cumdist = list(itertools.accumulate(weights))
    # choose sel_sz number of choices with probability proportional to weights
    for i in range(sel_sz):
      # TODO: Check this?
      x = random.random()*cumdist[-1]
      selected_pop.append(choices[bisect.bisect(cumdist, x)])
    return selected_pop


  def initialization(self):
    chrom_length = len(self.all_features)
    # Initial population contains 10*N chromosomes/strings
    return [get_random_chromosome(chrom_length) 
        for i in range(10*chrom_length)]


