import numpy as np
from scipy.stats import truncnorm
from math import sqrt
from typing import Any
from copy import deepcopy


class EpochDE:

    def __init__(self, size: int = 100, dim: int = 0):

        self.p_inf = 0.0
        self.p_sup = 1.0
        self.s_inf = 1. / sqrt(size)
        self.s_sup = 2.0
        self.gamma = 0.9

        # generate init values
        self.size = size
        self.dim = dim
        self.population = np.array([truncnorm.rvs(a=0, b=1, size=self.dim) for _ in range(size)])
        self.best_idx = 0

        # initialization of parameters of mutation and crossover
        self.p = np.array([np.random.normal(self.p_inf, self.p_sup) for _ in range(self.dim)])
        self.s = np.array([np.random.normal(self.s_inf, self.s_sup) for _ in range(self.dim)])
        self.prev_variations = np.array([self.population[:, i].var() for i in range(self.dim)])
        self.current_variations = np.array([])
        self.ro = np.array([])
        self.flag = False

    def de_epoch(self, models: np.array, weight_scores: np.array, fitness_mse: Any, fitness_wmse: Any):
        indices = list(range(self.size))

        trial_generation = []
        trial_target_values = []

        if self.current_variations.size != 0:
            self.ro = np.array([self.gamma * (var_prev / var) for var_prev, var in zip(self.prev_variations,
                                                                                       self.current_variations)])

        for j, individual in enumerate(self.population):

            # choosing parents and mutation
            choice = [j]
            while j in choice:
                choice = np.random.choice(indices, 3, replace=False)

            ind_c, ind_a, ind_b = choice
            # mutation
            c_mutant = self.mutate(ind_a, ind_b, ind_c)

            # recounting recombination probabilities
            if self.current_variations.size != 0:
                self.recount_probabilities()
                self.flag = not self.flag

            t_child = self.crossing(individual, c_mutant)
            trial_generation.append(t_child)

        # here will be wmse and mse counting block
        fitness_wmse.weights = np.vstack(trial_generation)
        for model in models:
            fitness_wmse.Evaluate(model)
        wmse = np.array([model.fitness.copy() for model in models])

        for i in range(len(fitness_wmse.weights)):
            weight_column = wmse[:, i]
            best_model_idx = np.argmin(weight_column)
            models[best_model_idx].fitness = 0
            fitness_mse.Evaluate(models[best_model_idx])
            trial_target_values = np.append(trial_target_values, deepcopy(models[best_model_idx].fitness))

        # selection. Вот здесь над ней надо подумать!!!!
        for j, (previous, current) in enumerate(zip(weight_scores, trial_target_values)):
            if previous > current:
                weight_scores[j] = current
                self.population[j] = trial_generation[j]
                if current < weight_scores[self.best_idx]:
                    self.best_idx = j

        # new variance counting
        if self.current_variations.size != 0:
            for k in range(self.dim):
                self.prev_variations[k] = self.current_variations[k]
                self.current_variations[k] = self.population[:, k].var()
        else:
            self.current_variations = np.array([self.population[:, k].var() for k in range(self.dim)])

    @staticmethod
    def round_value(value: float) -> float:
        return 0 if value < 0 else 1

    def mutate(self, ind_a: int, ind_b: int, ind_c: int) -> np.array:

        # simple mutation
        mutant = np.array([])
        for i, (gen_c, gen_a, gen_b) in enumerate(zip(self.population[ind_c], self.population[ind_a],
                                                      self.population[ind_b])):
            value = gen_c + self.s[i] * (gen_a - gen_b)
            mutant = np.append(mutant, value if 0 <= value <= 1 else EpochDE.round_value(value))
        return mutant

    def crossing(self, first_parent: np.array, second_parent: np.array) -> np.array:
        # then, according to the received events, we transfer genes
        child = np.array([first_parent[ind] if self.p[ind] else second_parent[ind]
                          for ind in range(self.dim)])
        return child

    def recount_probabilities(self):
        NP = len(self.ro)
        if self.flag:

            for k in range(NP):
                condition = NP * (self.ro[k] - 1) + self.p[k] * (2 - self.p[k])
                if condition >= 0:
                    self.s[k] = sqrt(condition/(2 * NP * self.p[k]))
                else:
                    self.s[k] = self.s_inf

        else:

            for k in range(NP):
                condition = self.ro[k]
                if condition >= 1:
                    self.p[k] = -(NP * self.s[k]**2 - 1) + sqrt((NP * self.s[k]**2 - 1)**2 - NP * (1 - self.ro[k]))
                else:
                    self.p[k] = self.p_inf
