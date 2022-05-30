from copy import deepcopy
from typing import Any

from simplegp.Selection import Selection
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness

from sr_for_itgp import EpochSR
from de_for_itgp import EpochDE
from models_serialization import load_models, load_weights, save_models, save_weights, MODELS_SAVEFILE, WEIGHTS_SAVEFILE
from models_serialization import readable_output_weights, readable_output_models
from models_serialization import FILE_SUFFIX, MODELS_FOR_CHECK, WEIGHTS_FOR_CHECK

# constants for optimization

GENERATIONS_SIZE = 100  # number of algorithm iterations
MODELS_POP_SIZE = 300  # model-tree population size (was 512 according to the article)
WEIGHTS_POP_SIZE = 50  # size of weight vectors population
TOP_MODELS_SIZE = WEIGHTS_POP_SIZE // 2
NOTES_NAME = "notes_iter"
# (it should be increased by factor of 10, instead of 3, because variables arent 10, as in Friedman function, but 120+)
D_NUM = 30  # number of measurements on which the tournament selection is carried out
T_NUM = 4  # size of tournament, in which the candidates are compared with each other
VARIABLE_SYMBOL = 'x'  # auxiliary symbol to denote variables by numbers in the sympy interpolation


def ITGP(x_source: np.array, y_source: np.array, x_target: np.array, y_target: np.array, dirname: str, model_ind: int,
         preload_models: bool = False, fileid: int = -1):

    weights_size = WEIGHTS_POP_SIZE
    models_size = MODELS_POP_SIZE
    top_models_size = TOP_MODELS_SIZE
    weights_dim = len(x_source)
    models_dim = len(x_source[0])
    de_estimator = EpochDE(weights_size, weights_dim)
    weights = de_estimator.population

    fitness_function = SymbolicRegressionFitness(x_source, y_source, use_weights=True, weights=weights)  # wmse all
    fitness_function_target = SymbolicRegressionFitness(X_train=x_target, y_train=y_target)  # mse all models
    fitness_function_source = SymbolicRegressionFitness(X_train=x_source, y_train=y_source)  # mse top models

    # for models evaluating
    srgp_estimator = EpochSR(dim=models_dim, fitness_function=fitness_function, pop_size=models_size, max_tree_size=320,
                             crossover_rate=0.9, mutation_rate=0.1, op_mutation_rate=0.1, min_height=3,
                             initialization_max_tree_height=10,
                             functions=[AddNode(), SubNode(), MulNode(), DivNode(), SqrtNode()])

    if preload_models:
        models = load_models("models_weights_info/models" + str(fileid) + FILE_SUFFIX, MODELS_POP_SIZE)
        weights = load_weights("models_weights_info/weights" + str(fileid) + FILE_SUFFIX)
        adjust_models_dimensions(models, srgp_estimator)
        # доделать стоит, если необходимо подгружать веса с разными размерностями
        # (грубо говоря, сначала датасет тренировочный имел один размер, а с нового запуска другой)
        adjust_weights_dimensions(weights, fitness_function)

    for i in range(GENERATIONS_SIZE):  # main cycle
        print("Epoch #{}".format(i))
        # weighted the models on the raw data and weights from the previous step
        models = srgp_estimator.population
        wmse = np.array([model.fitness.copy() for model in models])

        # Here we will select the best TOP_MODELS_SIZE models for evaluation on the target data
        re_eval_models = Selection.models_selection_wmse(models, wmse, fitness_function_target)
        trail_weights = np.array([model.fitness.copy() for model in re_eval_models])
        de_estimator.de_epoch(deepcopy(re_eval_models), trail_weights,
                              fitness_mse=fitness_function_target, fitness_wmse=fitness_function)

        # choosing top tp models from target evaluated
        top_models = Selection.models_selection_mse(re_eval_models, top_models_size)
        D = Selection.roulette_wheel_selection(trail_weights, D_NUM)
        for model in top_models:
            fitness_function.Evaluate(model)

        # rest of p-tp models are tournament-selected here
        other_models = Selection.TournamentSelect(models, models_size - top_models_size, tournament_size=4,
                                                  D=D, eps=1e-3)
        fitness_function.weights = de_estimator.population
        srgp_estimator.population = top_models + other_models
        srgp_estimator.sr_epoch()

    save_weights(dirname + WEIGHTS_SAVEFILE + str(model_ind) + FILE_SUFFIX, weights)
    save_models(dirname + MODELS_SAVEFILE + str(model_ind) + FILE_SUFFIX, models)
    # getting mse values for each model in ending population (for readable fitness output)

    readable_output_weights(dirname + WEIGHTS_FOR_CHECK + str(model_ind) + FILE_SUFFIX, weights)
    readable_output_models(dirname + MODELS_FOR_CHECK + str(model_ind) + FILE_SUFFIX, models,
                           fitness_target=fitness_function_target, fitness_source=fitness_function_source)
    for model in models:
        fitness_function_target.Evaluate(model)
    res_model = choose_best_model(models, fitness_source=fitness_function_source,
                                  fitness_target=fitness_function_target)
    res_model.append(model_ind)
    return res_model


def choose_best_model(models: list, fitness_source: SymbolicRegressionFitness,
                      fitness_target: SymbolicRegressionFitness) -> list:
    best_trg, best_src, best_model = np.inf, np.inf, 0
    for model in models:
        fitness_target.Evaluate(model)
        model_trg_fitness = model.fitness
        fitness_source.Evaluate(model)
        model_src_fitness = model.fitness
        if (np.sqrt(model_src_fitness) <= np.sqrt(model_trg_fitness) or
            (np.fabs(np.sqrt(model_src_fitness) - np.sqrt(model_trg_fitness))
             <= 5.)) and best_trg >= model_trg_fitness:
            best_trg = model_trg_fitness
            best_src = model_src_fitness
            best_model = model
    return [deepcopy(best_model), best_src, best_trg]


def adjust_weights_dimensions(weights_pop: np.array, fitness_function: SymbolicRegressionFitness):
    # not really a working method, it should be finalized if necessary and after discussion
    curr_size = len(fitness_function.weights)
    preload_size = len(weights_pop)

    if preload_size < curr_size:
        for i in range(preload_size):
            fitness_function.weights[i] = weights_pop[i]
    elif preload_size > curr_size:
        for i in range(curr_size):
            fitness_function.weights[i] = weights_pop[i]
    else:
        fitness_function.weights = weights_pop


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
