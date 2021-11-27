import numpy as np
from random import choices, randint, choice
import math
import collections
from enum import Enum 

IndividualType = Enum('IndividualType', 'Binary Float')
ReproductionType = Enum('ReproductionType', 'Crossover Average')

class IslandGeneticAlgorithm:
  
  def __init__(
      self, 
      n_epochs, 
      individual_type, 
      array_size, 
      n_islands, 
      n_individuals, 
      n_parents, 
      fitness_functions, 
      mutation_rate, 
      step, 
      reproduction_type = ReproductionType.Crossover
    ):
    """
    The constructor accept multiple parameters allowing to parameterize the algorithm

    :param n_epochs: number of epochs to train the IslandGeneticAlgorithm
    :param individual_type: type IndividualType, Binary if the problem is a binary interger programming
                            problem, otherwise Float
    :param array_size: size of the solution array
    :param n_islands: number of parallel genetic algorithm to train
    :param n_individuals: number of individuals in the population of each island
    :param n_parents: number of parents to take into account
    :param fitness_functions: function that determines the fittest individuals. There can be multiple
                              fitness function. Format: [{'function': func1, 'parameters': params1}, ...]
    :param mutation_rate: the probability of an element to be randomly mutated
    :param step: the size of the mutation if individual_type == IndividualType.Float
    :param reproduction_type: The crossover method, between ReproductionType.Crossover and ReproductionType.Average
                              If individual_type == IndividualType.Binary, the reproduction_type must be
                              crossover
    """
    self.n_epochs = n_epochs
    self.individual_type = individual_type
    self.reproduction_type = reproduction_type
    if individual_type == IndividualType.Binary:
      assert reproduction_type == ReproductionType.Crossover
      
    self.array_size = array_size
    self.n_islands = n_islands
    self.n_individuals = n_individuals
    self.n_parents = n_parents
    assert n_parents < n_individuals
    
    self.n_offsprings = n_individuals - n_parents
    self.mutation_rate = mutation_rate
    self.step = step
    self.fitness_functions = fitness_functions
    
    self.islands = []
    for _ in range(self.n_islands):
      self.islands.append(np.zeros((self.n_individuals, self.array_size)))
      
    self.parent_weights = []
    for i in range(self.n_parents):
      p = 0.1
      self.parent_weights.append(p * (1 - p)**i + ((1 - p)**self.n_parents) / self.n_parents)
      
    self.best_solutions = []
    self.best_solution = []
    
  def evaluate_individuals(self):
    """
    Evaluate each individual of each island for each fitness function

    :return: A list of the fitness score for every individuals
    """
    all_islands_results = []
    for island in self.islands:
      island_results = collections.defaultdict(list)
      for i in range(len(island)):
        for function in (self.fitness_functions):
          island_results[i].append(function['function'](island[i], function['parameters']))
      all_islands_results.append(island_results)
    return all_islands_results
  
  def pareto_front(self, costs):
    """
    Evaluate each individual of each island for each fitness function.
    It finds the pareto front that maximize the fitness score for every fitness function.

    :param costs: the fitness scores for each individual of an island
    :return: A mask of the pareto front
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
      if is_efficient[i]:
        is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)
        is_efficient[i] = True
        
    is_efficient_mask = np.zeros(costs.shape[0], dtype = bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask
  
  def select_best_individuals(self, n_individuals):
    """
    Select the n_individuals best individuals with the pareto front

    :param n_individuals: the number of individuals to select
    :return: The best individuals for each islands
    """
    islands_best_individuals = []
    all_islands_results = self.evaluate_individuals()
    
    for i in range(self.n_islands):
      best_individuals = np.array([]).reshape(0, self.array_size)
      cost = np.array(list(all_islands_results[i].values()))
      current_individuals = self.islands[i].copy()

      while (len(best_individuals) < n_individuals):
        mask = self.pareto_front(cost)
        if(best_individuals.shape[0] + sum(mask) > n_individuals):
          potential_individuals = current_individuals[mask]
          new_individual = potential_individuals[:(n_individuals - best_individuals.shape[0])]
          best_individuals = np.concatenate((best_individuals, new_individual))
        else:
          best_individuals = np.concatenate((best_individuals, current_individuals[mask]))
          current_individuals = current_individuals[~mask]
          cost = cost[~mask]

      islands_best_individuals.append(best_individuals)
    return islands_best_individuals
  
  def add_mutation(self):
    """
    Add random mutation to individual element with a probability of self.mutation_rate.
    If individual_type == IndividualType.Binary, the mutation toggle an element.
    If individual_type == IndividualType.Float, the mutation is a change of size step.
    """
    for island in self.islands:
      for _ in range(math.floor(self.n_individuals * self.array_size * self.mutation_rate)):
        random_individual = randint(0, island.shape[0] - 1)
        random_project = randint(0, self.array_size - 1)
        if self.individual_type == IndividualType.Binary:
          island[random_individual][random_project] = abs(island[random_individual][random_project] - 1)
        elif self.individual_type == IndividualType.Float:
          step = choice([-1, 1]) * self.step
          if (step < 0) and (island[random_individual][random_project] + step < 0):
            step *= -1
          island[random_individual][random_project] += step
      
  def update_generation(self):
    """
    Create self.n_offsprings new individuals using crossover or average, depending on self.reproduction_type.
    It then add mutation and select the best self.n_individuals
    """
    best_individuals = self.select_best_individuals(self.n_parents)
    
    for i in range(self.n_islands):
      offsprings = []
      if self.reproduction_type == ReproductionType.Crossover:
        for _ in range(self.n_offsprings):
          n_crossovers = 2
          crossover_positions = sorted(choices(range(self.array_size), k = n_crossovers))
          crossover_positions.insert(0, 0)
          crossover_positions.append(self.array_size)
          random_individuals = choices(range(self.n_parents), weights = self.parent_weights, k = n_crossovers + 1)
          offspring = []
          for j in range(n_crossovers + 1):
            projects = best_individuals[i][random_individuals[j]][crossover_positions[j] : crossover_positions[j+1]].copy()
            offspring.extend(projects)
          offsprings.append(offspring)
      elif self.reproduction_type == ReproductionType.Average:
        for _ in range(self.n_offsprings):
          random_individuals = choices(range(self.n_parents), weights = self.parent_weights, k = 2)
          offspring = (best_individuals[i][random_individuals[0]] + best_individuals[i][random_individuals[1]]) / 2
          offsprings.append(offspring)
        
      offsprings = np.array(offsprings)
      self.islands[i] = np.concatenate((self.islands[i], offsprings))
    
    self.add_mutation()
    self.islands = self.select_best_individuals(self.n_individuals)
    
  def train(self):
    """
    Execute self.n_epochs generations and each 100 epochs, the best individual of the island i is 
    added to the island i+1
    """
    epoch_number = 0
    best_result = 0
    
    while(epoch_number < self.n_epochs):
      self.update_generation()
      
      results = []
      function = self.fitness_functions[0]
      for i in range(self.n_islands):
        island_result = function['function'](self.islands[i][0], function['parameters'])
        results.append(round(island_result, 4))
        if (island_result > best_result):
          best_result = island_result
          self.best_solutions.append(self.islands[i][0])
          print('Best solution updated')
      print('Epoch : ', epoch_number, ', Fitness: ', results)
       
      epoch_number += 1
      
      if (epoch_number % 100 == 0):
        print('Migration')
        for i in range(self.n_islands):
          self.islands[(i + 1) % self.n_islands][-1] = self.islands[i][0].copy()
  
  def evaluate(self):
    """
    Get the best solution of all according to the first fitness function.
    """
    results = {}
    function = self.fitness_functions[0]
    for i in range(len(self.best_solutions)):
      results[i] = function['function'](self.best_solutions[i], function['parameters'])

    sorted_results = sorted(results, key=lambda x: results[x], reverse=True)
    self.best_solution = self.best_solutions[sorted_results[0]]
    