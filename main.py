from ITGP import ITGP, choose_best_model
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from different_utils import my_cv, run_parallel_tests
from dataset_parsing import initial_parse_data_and_save, MERGED_DATASET, parse_valid, data_shuffle
from dataset_parsing import parse_x_y, parse_autumn_spring
from dataset_parsing import SEASON_KEY, GEO_ID_KEY, DATASET_SEASONS, SNP_KEYS, get_data_response
from models_serialization import load_models

import os
import re
import numpy as np
import pandas as pd
import time
import multiprocessing
from multiprocessing import Pool, Process, freeze_support
from copy import deepcopy
from sklearn.decomposition import PCA

TRAIN_SIZE = 2500
TEST_SIZE = 500
VALIDATION_SIZE = 1000
DAYS_PER_SNIP = 20  # число дней для предсказания погоды


def main():
    np.random.seed(3)

    redundant_column_name = "Unnamed: 0"
    response = get_data_response()  # parameter for prediction in future
    predictors = pd.read_csv(MERGED_DATASET, sep=";")
    del predictors[redundant_column_name]
    n_folds = 5
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
    # results_filename = "temp_results/cv_data_from_article.txt"
    # file = open(results_filename, "w")
    # cv_32 = my_cv(x_src, y_src, x_trg,
    #               y_trg, ITGP, n_folds, file)
    # mess = "CV result with Turkey data as source and Australia data as target: {}".format(cv_32[1])
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()

    x_numpy, y_numpy = x_valid.to_numpy(), y_valid.to_numpy().flatten()
    train_function = SymbolicRegressionFitness(X_train=x_numpy, y_train=y_numpy)
    file = open("temp_results/cv_short_results.txt", "w")
    # print("All best function, created during cross-vaidation: ")
    # for model_data in cv_32[0]:
    #     print("#######################################")
    #     file.write("#######################################\n")
    #     print("Best model in file 'models{0}.txt': {1}".format(model_data[3], model_data[0]))
    #     file.write("Best model in file 'models{0}.txt': {1}".format(model_data[3], model_data[0]) + "\n")
    #     print("Model fitness of source data: {}".format(model_data[1]))
    #     file.write("Model fitness of source data: {}".format(model_data[1]) + "\n")
    #     print("Model fitness of target data: {}".format(model_data[2]))
    #     file.write("Model fitness of target data: {}".format(model_data[2]) + "\n")
    #     train_function.Evaluate(model_data[0])
    #     print("Model fitness of validation data: {}".format(model_data[0].fitness))
    #     file.write("Model fitness of validation data: {}".format(model_data[0].fitness) + "\n")

    dirname = "models_weights_info/"
    # dirname = "yesterday_results/"
    files = os.listdir(dirname)
    model_files = []
    for file in files:
        if re.search("^models", file) != 0:
            model_files.append(file)
    best, best_filename = np.inf, ""
    populations_num = len(files) // 4
    for i in range(populations_num):
        print("Models filename: {}".format(model_files[i]))
        models = load_models(dirname + model_files[i], 300)
        for model in models:
            train_function.Evaluate(model)
            if np.sqrt(model.fitness) < best:
                best = np.sqrt(model.fitness)
                best_filename = model_files[i]
            print("Model fitness on validation data: {}".format(np.sqrt(model.fitness)))
    print("Best model fitness: {}".format(best))
    print("In file: {}".format(best_filename))


if __name__ == '__main__':
    freeze_support()

    # cross-validation function
    main()

    # тестовые значения, для отладки кода
    # source_size = 1000
    # target_size = 120
    # n_folds = 10
    # x_s, y_s = pd.read_csv("geo_datasets/data_geo_2.0.csv", sep=";"), \
    #        pd.read_csv("geo_datasets/data_geo_2.0_response.csv", sep=";")
    # x_t, y_t = pd.read_csv("geo_datasets/data_geo_3.0.csv", sep=";"), \
    #            pd.read_csv("geo_datasets/data_geo_3.0_response.csv", sep=";")
    # x_numpy, y_numpy = x_s.to_numpy(), y_s.to_numpy().flatten()
    # x_t_numpy, y_t_numpy = x_t.to_numpy(), y_t.to_numpy().flatten()
    # source_x, source_y = parse_x_y(x_numpy, y_numpy, source_size)
    # target_x, target_y = parse_x_y(x_t_numpy, y_t_numpy, target_size)
    # start_time = time.time()
    # res = ITGP(source_x, source_y, target_x, target_y, preload_models=False, fileid=75)
    # print("Function ITGP worktime: {}".format(time.time() - start_time))
    # expr = res[0].GetHumanExpression()
    # print("Best model: {}".format(expr))
    # print("Function target fitness: {}".format(res[2]))
    # print("Function source fitness: {}".format(res[1]))
    # run_parallel_tests(source_x, source_y, target_x, target_y, source_size, target_size)
