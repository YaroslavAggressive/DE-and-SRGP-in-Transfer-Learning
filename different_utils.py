import numpy as np
from typing import Any
from pandas import DataFrame
from typing import Callable
from copy import deepcopy
import multiprocessing
from multiprocessing import Pool, Process
from dataset_parsing import parse_x_y
from ITGP import ITGP


def determination_coeff(y_true: np.array, y_predict: np.array):
    y_mean = np.mean(y_true)
    res = np.square(y_true - y_predict)
    tot = np.square(y_true - y_mean)
    return 1 - res / tot


def my_cv(x_df_src: DataFrame, y_df_src: DataFrame, x_df_trg: DataFrame, y_df_trg: DataFrame,
          method: Callable, dirname: str, n_folds: int, file: Any) -> list:
    numpy_x_src, numpy_y_src = x_df_src.to_numpy(), y_df_src.to_numpy().flatten()
    numpy_x_trg, numpy_y_trg = x_df_trg.to_numpy(), y_df_trg.to_numpy().flatten()
    x_src_for_cv = np.array_split(numpy_x_src, n_folds, axis=0)
    x_trg_for_cv = np.array_split(numpy_x_trg, n_folds, axis=0)
    y_src_for_cv = np.array_split(numpy_y_src, n_folds, axis=0)
    y_trg_for_cv = np.array_split(numpy_y_trg, n_folds, axis=0)
    cv_res = 0
    data = []
    # оставляю 2 ядра, чтобы ноут не помер
    with Pool(multiprocessing.cpu_count() - 2) as cv_pool:
        for i in range(n_folds):
            for j in range(n_folds):
                print("CV iteration # {}".format(str(i)))
                target_x, target_y = x_trg_for_cv[j], y_trg_for_cv[j]
                source_x, source_y = x_src_for_cv[i], y_src_for_cv[i]
                data.append((source_x, source_y, target_x, target_y, dirname, False))
        data = cv_pool.starmap(method, data)
        for i, model_data in enumerate(data):
            cv_res += model_data[2]
            print("Best model # {}".format(model_data[0].GetHumanExpression()))
            file.write("Best model # {}".format(model_data[0].GetHumanExpression()) + "\n")
            print("Model target fitness: {}".format(model_data[2]))
            file.write("Model target fitness: {}".format(model_data[2]) + "\n")
            print("Model source fitness: {}".format(model_data[1]))
            file.write("Model source fitness: {}".format(model_data[1]) + "\n")
            print("File index with all models information: {}".format(model_data[3]))
            file.write("File index with all models information: {}".format(model_data[3]) + "\n")
        print("Total error: {}".format(cv_res // n_folds))
        file.write("Total error: {}".format(cv_res // n_folds) + "\n")
    return [data, cv_res // n_folds]


def run_parallel_tests(x_source: np.array, y_source: np.array, x_target: np.array, y_target: np.array,
                       source_size: int, target_size: int):
    # method for creating model on chosen dataset configuration, given in this method
    seeds = list(range(20))
    with Pool(multiprocessing.cpu_count() - 2) as tests_pool:
        data = []
        for seed in seeds:
            np.random.seed(seed)
            x_s, y_s = parse_x_y(x_source, y_source, source_size)
            x_t, y_t = parse_x_y(x_target, y_target, target_size)
            data.append((x_s, y_s, x_t, y_t, False))
        results = tests_pool.starmap(ITGP, data)

    mean_fit_source, mean_fit_target = 0, 0
    for res in results:
        mean_fit_source += res[1]
        mean_fit_target += res[2]
    print("Mean best model fitness on source data: {}".format(mean_fit_source // len(results)))
    print("Mean best model fitness on target data: {}".format(mean_fit_target // len(results)))
    return [mean_fit_source, mean_fit_target]
