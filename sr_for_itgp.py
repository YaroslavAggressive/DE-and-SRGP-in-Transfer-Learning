import inspect
from sklearn.base import BaseEstimator
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Variation import Variation
from simplegp.Selection import Selection
from copy import deepcopy
from numpy.random import random, randint
from multiprocessing import Pool, Process, Queue


class EpochSR(BaseEstimator):

    def __init__(self, dim,
                 fitness_function,
                 functions,
                 use_erc=True,
                 pop_size=500,
                 crossover_rate=0.5,
                 mutation_rate=0.5,
                 op_mutation_rate=0.0,
                 max_evaluations=-1,
                 min_height=2,
                 initialization_max_tree_height=4,
                 max_tree_size=100,
                 max_features=-1,
                 tournament_size=4):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        for arg, val in values.copy().items():
            setattr(self, arg, val)

        self.terminals = []
        # if self.use_erc:  # пока не понял, понадобится ли это
        #     self.terminals.append(EphemeralRandomConstantNode())
        n_features = dim
        for i in range(n_features):
            self.terminals.append(FeatureNode(i))

        # initializing population of models
        self.population = []
        curr_max_depth = self.min_height
        init_depth_interval = self.pop_size / (self.initialization_max_tree_height - self.min_height + 1)
        next_depth_interval = init_depth_interval

        for i in range(self.pop_size):
            if i >= next_depth_interval:
                next_depth_interval += init_depth_interval
                curr_max_depth += 1

            t = Variation.GenerateRandomTree(self.functions, self.terminals, curr_max_depth, curr_height=0,
                                             method='grow' if np.random.random() < .5 else 'full',
                                             min_height=self.min_height)
            self.fitness_function.Evaluate(t)
            self.population.append(t)

    def sr_epoch(self):

        trial = []
        for i in range(self.pop_size):
            elem = deepcopy(self.population[i])
            variation_happened = False
            while not variation_happened:
                if random() < self.crossover_rate:
                    elem = Variation.SubtreeCrossover(elem, self.population[randint(self.pop_size)])
                    variation_happened = True
                if random() < self.mutation_rate:
                    elem = Variation.SubtreeMutation(elem, self.functions, self.terminals,
                                                     max_height=self.initialization_max_tree_height,
                                                     min_height=self.min_height)
                    variation_happened = True
                if random() < self.op_mutation_rate:
                    elem = Variation.OnePointMutation(elem, self.functions, self.terminals)
                    variation_happened = True

            # check offspring meets constraints
            invalid_offspring = False
            if len(elem.GetSubtree()) > self.max_tree_size > -1:
                invalid_offspring = True
            elif elem.GetHeight() < self.min_height:
                invalid_offspring = True
            elif self.max_features > -1:
                features = set()
                for n in elem.GetSubtree():
                    if hasattr(n, 'id'):
                        features.add(n.id)
                if len(features) > self.max_features:
                    invalid_offspring = True
            if invalid_offspring:
                del elem
                elem = deepcopy(self.population[i])
            else:
                self.fitness_function.Evaluate(elem)

            trial.append(elem)

        # Selection (in this program it is implemented in main cycle)
        self.population = trial
        # self.population = Selection.TournamentSelect(PO, self.pop_size, tournament_size=self.tournament_size)
