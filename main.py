from ITGP import ITGP, choose_best_model
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from different_utils import my_cv, run_parallel_tests
from dataset_parsing import initial_parse_data_and_save, MERGED_DATASET, parse_valid, data_shuffle
from dataset_parsing import parse_x_y, parse_autumn_spring
from dataset_parsing import SEASON_KEY, GEO_ID_KEY, DATASET_SEASONS, SNP_KEYS, get_data_response
from models_serialization import load_models
from model_research import main_research

import os
import re
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from multiprocessing import freeze_support
from copy import deepcopy
from sklearn.decomposition import PCA

TRAIN_SIZE = 2500
TEST_SIZE = 500
VALIDATION_SIZE = 1000
DAYS_PER_SNIP = 20  # число дней для предсказания погоды


def main(seed: int, iter_name: int):
    test_path = "models_weights_info/iter{}".format(iter_name)
    if os.path.exists(test_path):
        raise FileExistsError("Directory with current number already exists in this dir, change iter_num value")
    os.mkdir(test_path)
    np.random.seed(seed)

    redundant_column_name = "Unnamed: 0"
    response = get_data_response()  # parameter for prediction in future
    predictors = pd.read_csv(MERGED_DATASET, sep=";")
    del predictors[redundant_column_name]
    n_folds = 4
    x_src, y_src, x_trg, y_trg = parse_autumn_spring(predictors, response)
    x_src, y_src = data_shuffle(x_src, y_src)
    x_trg, y_trg = data_shuffle(x_trg, y_trg)
    src_size = x_src.shape[0]
    trg_size = src_size // 5
    valid_size = x_trg.shape[0] - trg_size

    x_valid, y_valid = parse_valid(x_trg, y_trg, valid_size)
    valid_indices = x_valid.index
    all_trg_indices = np.unique(list(range(trg_size)))
    trg_no_valid_indices = np.setdiff1d(all_trg_indices, valid_indices)
    x_trg, y_trg = x_trg.loc[trg_no_valid_indices], y_trg.loc[trg_no_valid_indices]
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

    train_function = SymbolicRegressionFitness(X_train=x_valid.to_numpy(), y_train=y_valid.to_numpy().flatten())
    source_function = SymbolicRegressionFitness(X_train=x_src.to_numpy(), y_train=y_src.to_numpy().flatten())
    target_function = SymbolicRegressionFitness(X_train=x_trg.to_numpy(), y_train=y_trg.to_numpy().flatten())
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

    files = os.listdir(test_path)
    model_files = []
    for file in files:
        if not os.path.isdir(test_path + "/" + file) and re.search("^models", file) != 0:
            model_files.append(file)
    populations_num = len(files) // 4
    for i in range(populations_num):
        print("Models filename: {}".format(model_files[i]))
        models = load_models(test_path + "/" + model_files[i], 300)

        for model in models:
            target_function.Evaluate(model)
        best_model_data = choose_best_model(models, source_function, target_function)
        train_function.Evaluate(best_model_data[0])
        print("Best model validation fitness: {}".format(np.sqrt(best_model_data[0].fitness)))
        print("Best model source fitness: {}".format(np.sqrt(best_model_data[1])))
        print("Best model target fitness: {}".format(np.sqrt(best_model_data[2])))


if __name__ == '__main__':

    # cross-validation function
    # main(15, 19)

    # test seeds and models populations sizes
    mp.freeze_support()
    seeds = [3, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
    pop_sizes = [300, 300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 200, 300, 300, 300, 300, 300, 300, 300]
    pop_sizes_dict = {"iter" + str(i): pop_sizes[i] for i in range(len(pop_sizes))}
    seeds_dict = {"iter" + str(i): seeds[i] for i in range(len(seeds))}

    # total dataset and response reading
    redundant_column_name = "Unnamed: 0"
    response = get_data_response()  # parameter for prediction in future
    predictors = pd.read_csv(MERGED_DATASET, sep=";")
    del predictors[redundant_column_name]

    # making main research all over the fiven data
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

    research = main_research(x_df=predictors, y_df=response, all_iters=files_models,
                             sizes_per_iter=pop_sizes_dict, seeds_per_iter=seeds_dict)