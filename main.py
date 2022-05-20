from ITGP import ITGP, choose_best_model
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from different_utils import my_cv, run_parallel_tests
from dataset_parsing import initial_parse_data_and_save, MERGED_DATASET, parse_valid, data_shuffle
from dataset_parsing import parse_x_y, parse_per_key, parse_per_ageev_state, parse_per_season, parse_per_snp
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
    # for the same choice of target and c-resource datasets during training
    np.random.seed(3)

    # тестовые значения, для отладки кода
    source_size = 500
    target_size = 60
    n_folds = 5
    x_2, y_2 = pd.read_csv("geo_datasets/data_geo_2.0.csv", sep=";"), \
               pd.read_csv("geo_datasets/data_geo_2.0_response.csv", sep=";")
    x_3, y_3 = pd.read_csv("geo_datasets/data_geo_3.0.csv", sep=";"), \
               pd.read_csv("geo_datasets/data_geo_3.0_response.csv", sep=";")
    x_0, y_0 = pd.read_csv("geo_datasets/data_geo_0.0.csv", sep=";"), \
               pd.read_csv("geo_datasets/data_geo_0.0_response.csv", sep=";")

    # random rows permutation
    x_0, y_0 = data_shuffle(x_0, y_0)
    x_2, y_2 = data_shuffle(x_2, y_2)
    x_3, y_3 = data_shuffle(x_3, y_3)

    x_0_valid, y_0_valid = parse_valid(x_0, y_0, x_0.shape[0] // 10)
    x_2_valid, y_2_valid = parse_valid(x_2, y_2, x_2.shape[0] // 10)
    x_3_valid, y_3_valid = parse_valid(x_3, y_3, x_3.shape[0] // 10)
    indices_0, indices_2, indices_3 = np.unique(list(range(x_0.shape[0]))), np.unique(list(range(x_2.shape[0]))),\
                                      np.unique(list(range(x_3.shape[0])))
    valid_indices_0, valid_indices_2, valid_indices_3 = x_0_valid.index, x_2_valid.index, x_3_valid.index
    train_indices_0, train_indices_2, train_indices_3 = np.setdiff1d(indices_0, valid_indices_0),\
                                                        np.setdiff1d(indices_2, valid_indices_2),\
                                                        np.setdiff1d(indices_3, valid_indices_3)
    # x_numpy, y_numpy = x_2.to_numpy(), y_2.to_numpy()
    # x_t_numpy, y_t_numpy = x_3.to_numpy(), y_3.to_numpy()
    # source_x, source_y = parse_x_y(x_numpy, y_numpy, source_size)
    # target_x, target_y = parse_x_y(x_t_numpy, y_t_numpy, target_size)

    # cv for geo_id 3.0 (src) и 2.0 (trg)
    # results_filename = "temp_results/cv_geo_id_all.txt"
    # file = open(results_filename, "w")
    # cv_32 = my_cv(x_3.iloc[train_indices_3], y_3.iloc[train_indices_3], x_2.iloc[train_indices_2],
    #               y_2.iloc[train_indices_2], ITGP, n_folds, file)
    # mess = "CV result with geo data 2.0 as source and 3.0 as target: {}".format(cv_32[1])
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()

    # cv for geo_id for 0.0 (src) and 3.0 (trg)
    # file = open(results_filename, "a")
    # cv_no_snp_6 = my_cv(x_0.iloc[train_indices_0], y_0.iloc[train_indices_0],
    # x_3.iloc[train_indices_3], y_3.iloc[train_indices_3], ITGP, n_folds, file)
    # mess = "CV result with geo data 0.0 as source and 3.0 as target: {}".format(cv_no_snp_6)
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()
    #
    # # cv for geo_id for 2.0 (src) and 0.0 (trg)
    # file = open(results_filename, "a")
    # cv_no_snp_6 = my_cv(x_2.iloc[train_indices_2], y_2.iloc[train_indices_2], x_0.iloc[train_indices_0],
    # y_0.iloc[train_indices_0], ITGP, n_folds, file)
    # mess = "CV result with geo data 2.0 as source and 0.0 as target: {}".format(cv_no_snp_6)
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()
    #
    # # cv for geo_id for 0.0 (src) and 2.0 (trg)
    # file = open(results_filename, "a")
    # cv_no_snp_6 = my_cv(x_0.iloc[train_indices_0], y_0.iloc[train_indices_0], x_2.iloc[train_indices_2],
    # y_2.iloc[train_indices_2], ITGP, n_folds, file)
    # mess = "CV result with geo data 0.0 as source and 2.0 as target: {}".format(cv_no_snp_6)
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()

    # cv for geo_id for 3.0 (src) and 0.0 (trg)
    # file = open(results_filename, "a")
    # cv_30 = my_cv(x_3.iloc[train_indices_3], y_3.iloc[train_indices_3], x_0.iloc[train_indices_0],
    #                     y_0.iloc[train_indices_0], ITGP, n_folds, file)
    # mess = "CV result with geo data 3.0 as source and 2.0 as target: {}".format(cv_30[1])
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()
    #
    # # cv for geo_id for 0.0 (src) and 3.0 (trg)
    # cv_03 = my_cv(x_0.iloc[train_indices_0], y_0.iloc[train_indices_0], x_3.iloc[train_indices_3],
    #                     y_3.iloc[train_indices_3], ITGP, n_folds, file)
    # file = open(results_filename, "a")
    # mess = "CV result with geo data 3.0 as source and 0.0 as target: {}".format(cv_03[1])
    # print(mess)
    # print("\n")
    # file.write(mess + "\n")
    # file.write("\n")
    # file.close()

    frames, frames_responses = [x_0_valid, x_2_valid, x_3_valid], [y_0_valid, y_2_valid, y_3_valid]
    x_valid, y_valid = pd.concat(frames), pd.concat(frames_responses)
    x_numpy, y_numpy = x_valid.to_numpy(), y_valid.to_numpy().flatten()
    train_function = SymbolicRegressionFitness(X_train=x_numpy, y_train=y_numpy)

    # dirname = "models_weights_info/"
    dirname = "yesterday_results/"
    files = os.listdir(dirname)
    model_files = []
    for file in files:
        if re.search("^models", file) != 0:
            model_files.append(file)
    best = np.inf
    populations_num = len(files) // 4
    for i in range(populations_num):
        print("Models filename: {}".format(model_files[i]))
        models = load_models(dirname + model_files[i], 300)
        for model in models:
            # train_function.Evaluate(model)
            if np.sqrt(model.fitness) < best:
                best = np.sqrt(model.fitness)
            print("Model fitness on validation data: {}".format(np.sqrt(model.fitness)))
    print("Best model fitness: {}".format(best))


if __name__ == '__main__':
    freeze_support()

    # np.random.seed(9411223)
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
