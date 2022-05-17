from copy import deepcopy
from typing import Any

from simplegp.Selection import Selection
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness

from sr_for_itgp import EpochSR
from de_for_itgp import EpochDE
from models_serialization import load_models, load_weights, save_models, save_weights, MODELS_SAVEFILE, WEIGHTS_SAVEFILE
from models_serialization import readable_output_weights, readable_output_models, MODELS_FOR_CHECK, WEIGHTS_FOR_CHECK
from population import Population

# constants for optimization

SEED = 1  # so that the results of the method can be reconstructed
GENERATIONS_SIZE = 100  # число поколений прогона данных
MODELS_POP_SIZE = 100  # размер популяции моделей-деревьев
WEIGHTS_POP_SIZE = 60  # размер популяции весовых векторов
TOP_MODELS_SIZE = WEIGHTS_POP_SIZE // 2
D_NUM = WEIGHTS_POP_SIZE // 6  # число измерений, по которым проводится турнирная селекция
T_NUM = 4  # размер турнира, в котором между собой сравниваются кандидаты
DAYS_PER_SNIP = 20  # число дней для предсказания погоды
VARIABLE_SYMBOL = 'x'  # вспомогательный символ для обозначения переменных по номерам в интерперетации sympy


def ITGP(x_source: np.array, y_source: np.array, x_train: np.array, y_train: np.array, preload_models: bool = False):

    # np.random.seed(SEED)

    weights_size = WEIGHTS_POP_SIZE
    models_size = MODELS_POP_SIZE
    top_models_size = TOP_MODELS_SIZE
    weights_dim = len(x_source)
    models_dim = len(x_source[0])
    de_estimator = EpochDE(weights_size, weights_dim)
    population = de_estimator.population

    # for wmse, вроде поменял, но еще надо проверить, насколько корректно
    fitness_function = SymbolicRegressionFitness(x_source, y_source, use_weights=True, weights=population.individuals)
    fitness_function_target = SymbolicRegressionFitness(x_train, y_train)  # for mse

    # for models evaluating
    # теперь crossover_rate = 0.9, mutation_rate И op_mutation_rate = 0.1 согласно статье по ITGP
    srgp_estimator = EpochSR(dim=models_dim, fitness_function=fitness_function, pop_size=models_size, max_tree_size=100,
                             crossover_rate=0.9, mutation_rate=0.1, op_mutation_rate=0.1, min_height=2,
                             initialization_max_tree_height=4,
                             # functions=[AddNode(), SubNode(), MulNode(), DivNode(), LogNode(), CosNode(), SinNode()])
                             functions=[AddNode(), SubNode(), MulNode(), DivNode(), SinNode(), CosNode()])
                             # functions=[AddNode(), SubNode(), MulNode()])

    if preload_models:
        models = load_models(MODELS_SAVEFILE, MODELS_POP_SIZE)
        weights = load_weights(WEIGHTS_SAVEFILE)
        fitness_function.weights = weights.individuals
        srgp_estimator.population = models
        adjust_models_dimensions(models, srgp_estimator)
        adjust_weights_dimensions(weights, fitness_function)

    for i in range(GENERATIONS_SIZE):  # main cycle
        print("Epoch #{}".format(i))
        # weighed the models on the raw data and weights from the previous step
        models = srgp_estimator.population
        wmse = np.array([model.fitness for model in models])
        # Here we will select the best TOP_MODELS_SIZE models for evaluation on the target data
        re_eval_models = Selection.models_selection_wmse(models, wmse, fitness_function_target)
        # for model in re_eval_models:
        #     fitness_function_target.Evaluate(model)
        trail_weights = np.array([model.fitness for model in re_eval_models])
        de_estimator.de_epoch(deepcopy(re_eval_models), trail_weights, fitness_mse=fitness_function_target,
                              fitness_wmse=fitness_function)
        weights = de_estimator.population
        # choosing top tp models from target evaluated
        top_models = Selection.models_selection_mse(re_eval_models, top_models_size)
        for model in top_models:
            fitness_function.Evaluate(model)
        # здесь турнирно выбираются остальные p-tp моделей
        D = Selection.roulette_wheel_selection(wmse, D_NUM)  #
        other_models = Selection.TournamentSelect(models, models_size - top_models_size, tournament_size=4,
                                                  D=D, eps=1e-5)
        fitness_function.weights = weights.individuals
        srgp_estimator.population = top_models + other_models
        srgp_estimator.sr_epoch()

    save_weights(WEIGHTS_SAVEFILE, weights)
    save_models(MODELS_SAVEFILE, models)
    # getting mse values for each model in ending population (for readable fitness output)
    for model in models:
        model.fitness = 0
        fitness_function_target.Evaluate(model)
    readable_output_weights(WEIGHTS_FOR_CHECK, population.individuals)
    readable_output_models(MODELS_FOR_CHECK, models)

    return [population, models]


def choose_best_model(models: list) -> Any:
    best, best_model = np.inf, 0
    for model in models:
        if model.fitness <= best:
            best = model.fitness
            best_model = model
    return best_model


def adjust_weights_dimensions(weights_pop: Population, fitness_function: SymbolicRegressionFitness):
    curr_size = len(fitness_function.weights)
    preload_size = weights_pop.size

    if preload_size < curr_size:
        for i in range(preload_size):
            fitness_function.weights[i] = weights_pop.individuals[i]
    elif preload_size > curr_size:
        for i in range(curr_size):
            fitness_function.weights[i] = weights_pop.individuals[i]
    else:
        fitness_function.weights = weights_pop.individuals


def adjust_models_dimensions(models_pop: list, srgp_estimator: EpochSR):
    curr_size = len(srgp_estimator.population)
    preload_size = len(models_pop)

    if preload_size < curr_size:
        for i in range(preload_size):
            srgp_estimator.population[i] = models_pop[i]
    elif preload_size > curr_size:
        for i in range(curr_size):
            srgp_estimator.population[i] = models_pop[i]
    else:
        srgp_estimator.population = models_pop

