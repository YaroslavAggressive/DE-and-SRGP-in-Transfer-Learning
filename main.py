from ITGP import ITGP, choose_best_model
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from different_utils import my_cv
from dataset_parsing import get_meteo_data, get_ssm_data, MERGED_DATASET, parse_valid, data_shuffle, merge_data
from dataset_parsing import parse_x_y, parse_autumn_spring
from dataset_parsing import SEASON_KEY, GEO_ID_KEY, DATASET_SEASONS, SNP_KEYS, get_data_response
from models_serialization import load_models
from model_research import main_research, parse_data_per_iter

import os
import re
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import time
from multiprocessing import freeze_support
from copy import deepcopy
from sklearn.decomposition import PCA

DAYS_PER_SNIP = 20  # number of days to predict the weather


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
    x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(predictors, response, seed)
    if no_doy:
        del x_src["doy"]
        del x_trg["doy"]
        del x_valid["doy"]
    # cv for Turkey (source) and Australia (target) data
    results_filename = "temp_results/cv_data_from_article{}.txt".format(iter_name)
    file = open(results_filename, "w")
    cv_32 = my_cv(x_src, y_src, x_trg, y_trg, ITGP, test_path, n_folds, file)
    mess = "CV result with Turkey data as source and Australia data as target: {}".format(cv_32[1])
    print(mess)
    print("\n")
    file.write(mess + "\n")
    file.write("\n")
    file.close()

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


if __name__ == '__main__':
    freeze_support()
    # cross-validation function
    # main(4321, 46, no_doy=True)

    # test seeds and models populations sizes
    seeds = [3, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 222, 222, 199, 187, 1472,
             3333, 3333, 3333, 77, 77, 777, 1313, 1414, 1414, 1111, 1111, 1111, 1111]
    pop_sizes = [300, 300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 200, 300, 300, 300, 300, 300, 300, 300,
                 300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    pop_sizes_dict = {"iter" + str(i): pop_sizes[i] for i in range(len(pop_sizes))}
    seeds_dict = {"iter" + str(i): seeds[i] for i in range(len(seeds))}

    # total dataset and response reading
    response = get_data_response()  # parameter for prediction in future
    predictors = pd.read_csv("datasets/merged_weather_ssm_with_month_year_doy.csv", sep=";")
    # making main research all over the given data
    dirpath = "models_weights_info"
    dirs = os.listdir(dirpath)
    files_models = {directory: [] for directory in dirs}
    for directory in dirs:
        dir_files = os.listdir(dirpath + "/" + directory)
        for file in dir_files:
            if file.startswith("models"):
                files_models[directory].append(dirpath + "/" + directory + "/" + file)

    # a single run code to find the best models among the test values, after this in will be commented.
    # Run it, if you forked repo data
    # find_save_best_models(x_df=predictors, y_df=response, all_iters=files_models,
    #                       sizes_per_iter=pop_sizes_dict, seeds_per_iter=seeds_dict)

    # process_data = best_model_over_research(predictors.to_numpy(), response.to_numpy().flatten(), files_models,
    #                                         pop_sizes_dict)
    # best_population_checking(predictors, response, iter_seeds=seeds, sizes_=pop_sizes_dict)

    research = main_research(x_df=predictors, y_df=response, seeds_per_iter=seeds_dict, save=False, iter_left=0,
                             iter_right=20)
