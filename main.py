from ITGP import ITGP, choose_best_model
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Nodes.BaseNode import Node
from different_utils import my_cv
from dataset_parsing import get_meteo_data, get_ssm_data, MERGED_DATASET, parse_valid, data_shuffle, merge_data
from dataset_parsing import parse_x_y, parse_autumn_spring
from dataset_parsing import SEASON_KEY, GEO_ID_KEY, DATASET_SEASONS, SNP_KEYS, get_data_response
from models_serialization import load_models
from model_research import main_research, parse_data_per_iter, find_save_best_models
from sklearn.model_selection import cross_validate

import os
import re
import numpy as np
import pandas as pd
import datetime
from multiprocessing import freeze_support
from copy import deepcopy
from sklearn.decomposition import PCA
import pickle
from graphics import plot_response_change

DAYS_PER_SNIP = 20  # number of days to predict the weather


def count_test_scores(model: Node, x_test: np.array, y_test: np.array) -> list:
    fitness_func = SymbolicRegressionFitness(X_train=x_test, y_train=y_test)
    fitness_func.Evaluate(model)
    fitness_val = model.fitness
    model_out = model.GetOutput(x_test)
    errors = model_out - y_test
    return [errors, fitness_val]


def main(seed: int, iter_name: int, no_doy: bool = False):
    test_path = "models_weights_info/iter{}".format(iter_name)
    if os.path.exists(test_path):
        raise FileExistsError("Directory with current number already exists in this dir, change iter_num value")
    os.mkdir(test_path)
    np.random.seed(seed)

    # dataset loading and preprocessing
    response = get_data_response()  # parameter for prediction in future
    predictors = pd.read_csv(MERGED_DATASET, sep=";")
    n_folds = 4  # number of parts of dataset for cross-validation
    # здесь данные не только делятся на train/validation/test, но и перемешиваются
    x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(predictors, response, seed)
    if no_doy:
        del x_src["doy"]
        del x_trg["doy"]
        del x_valid["doy"]
    # cv for Turkey (source) and Australia (target) data
    results_filename = "temp_results/cv_data_from_article{}.txt".format(iter_name)
    file = open(results_filename, "w")
    cv_32 = my_cv(x_src, y_src, x_trg, y_trg, ITGP, test_path, n_folds, file)
    file.close()
    for_validation_save = "temp_results/best_data_models_and_validation_{}.txt".format(iter_name)
    with open(for_validation_save, "wb") as file:
        pickle.dump(cv_32, file)
    # вывод данных в консоль PyCharm для анализа полученных лучших моделей в каждой популяции
    train_function = SymbolicRegressionFitness(X_train=x_valid.to_numpy(), y_train=y_valid.to_numpy().flatten())
    file = open("temp_results/cv_short_results{}.txt".format(iter_name), "w")
    print("All best function, created during cross-validation: ")
    for model_data in cv_32[0]:
        print("#######################################")
        file.write("#######################################\n")
        print("Best model in file 'models{0}.txt': {1}".format(model_data[3], model_data[0]))
        file.write("Best model in file 'models{0}.txt': {1}".format(model_data[3], model_data[0]) + "\n")
        print("Model fitness of source data: {}".format(np.sqrt(model_data[1])))
        file.write("Model fitness of source data: {}".format(model_data[1]) + "\n")
        print("Model fitness of target data: {}".format(np.sqrt(model_data[2])))
        file.write("Model fitness of target data: {}".format(model_data[2]) + "\n")
        train_function.Evaluate(model_data[0])
        print("Model fitness of validation data: {}".format(np.sqrt(model_data[0].fitness)))
        file.write("Model fitness of validation data: {}".format(model_data[0].fitness) + "\n")
    file.close()

    best_model, best = None, np.inf
    for key, val in cv_32[1].items():
        tmp_model_sum = sum(val[1:])
        if tmp_model_sum <= best:
            best = tmp_model_sum
            best_model = deepcopy(val[0])
    print("Best model: {}".format(best_model[0].GetHumanExpression()))
    best_vals = count_test_scores(model=best_model[0], x_test=x_valid.to_numpy(), y_test=y_valid.to_numpy().flatten())
    plot_response_change(x_true=list(response.to_numpy().flatten()),
                         x_prediction=list(best_model.GetOutput(predictors.to_numpy())),
                         path_file="best_model_graph.jpg")

    print("Model fitness: {}".format(best_vals[1]))
    return best_vals


if __name__ == '__main__':
    freeze_support()
    print("Program started at: {}".format(datetime.datetime.now()))
    # cross-validation function
    main(10, 63, no_doy=True)

    # test seeds and models populations sizes
    # seeds = [3, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 222, 222, 199, 187, 1472,
    #          3333, 3333, 3333, 77, 77, 777, 1313, 1414, 1414, 1111, 1111, 1111, 1111, 10, 10, 123, 123, 1234, 4321,
    #          4321, 4321, 4321, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # pop_sizes = [300, 300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 200, 300, 300, 300, 300, 300, 300, 300,
    #              300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
    #              300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 400, 400, 500, 600]
    # pop_sizes_dict = {"iter" + str(i): pop_sizes[i] for i in range(len(pop_sizes))}
    # seeds_dict = {"iter" + str(i): seeds[i] for i in range(len(seeds))}
    #
    # # total dataset and response reading
    # response = get_data_response()  # parameter for prediction in future
    # predictors = pd.read_csv("datasets/merged_weather_ssm_with_month_year_doy.csv", sep=";")
    # # making main research all over the given data
    # dirpath = "models_weights_info"
    # dirs = os.listdir(dirpath)
    # files_models = {directory: [] for directory in dirs}
    # for directory in dirs:
    #     dir_files = os.listdir(dirpath + "/" + directory)
    #     for file in dir_files:
    #         if file.startswith("models"):
    #             files_models[directory].append(dirpath + "/" + directory + "/" + file)
    #
    # # a single run code to find the best models among the test values, after this in will be commented.
    # # Run it, if you forked repo data
    # # find_save_best_models(x_df=predictors, y_df=response, all_iters=files_models,
    # #                       sizes_per_iter=pop_sizes_dict, seeds_per_iter=seeds_dict, iters_left=1, iters_right=56)
    #
    # # process_data = best_model_over_research(predictors.to_numpy(), response.to_numpy().flatten(), files_models,
    # #                                         pop_sizes_dict)
    # # best_population_checking(predictors, response, iter_seeds=seeds, sizes_=pop_sizes_dict)
    #
    # research = main_research(x_df=predictors, y_df=response, seeds_per_iter=seeds_dict, save=True, iter_left=1,
    #                          iter_right=56, draw=False)

    # seed = 11
    # np.random.seed(seed)
    # response = get_data_response()  # parameter for prediction in future
    # predictors = pd.read_csv("datasets/merged_weather_ssm.csv", sep=";")
    # x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(predictors, response, seed=seed)
    # del x_trg['doy']
    # del x_src['doy']
    # del x_valid['doy']
    # res = ITGP(x_src.to_numpy(), y_src.to_numpy().flatten(), x_trg.to_numpy(), y_trg.to_numpy().flatten(), "test_iter",
    #            0, False, 1)
