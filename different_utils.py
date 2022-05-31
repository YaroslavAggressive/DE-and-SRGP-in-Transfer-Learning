import numpy as np
from numpy.random import default_rng
from typing import Any
from pandas import DataFrame
import pandas as pd
from typing import Callable
from copy import deepcopy
import multiprocessing
from multiprocessing import Pool, Process
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from dataset_parsing import parse_x_y
from ITGP import ITGP
from graphics import plot_cv_hist

ITERS_NUM = 50  # пока 10 для проверки и графиков и вообще показательности работы проги


def determination_coeff(y_true: np.array, y_predict: np.array):
    y_mean = np.mean(y_true)
    res = np.square(y_true - y_predict)
    tot = np.square(y_true - y_mean)
    return 1 - res / tot


def split_folds(x_df: DataFrame, y_df: DataFrame, n_folds: int) -> list:
    folds_x, folds_y = [], []
    df_size = x_df.shape[0]
    fold_size = df_size // n_folds
    total_indices = x_df.index
    for i in range(n_folds):
        tmp_indices = np.random.choice(total_indices, size=fold_size, replace=False)
        tmp_x, tmp_y = x_df.loc[tmp_indices, :], y_df.loc[tmp_indices, :]
        total_indices = np.setdiff1d(total_indices, tmp_indices)
        folds_x.append(tmp_x)
        folds_y.append(tmp_y)
    return [folds_x, folds_y]


def my_cv(x_df_src: DataFrame, y_df_src: DataFrame, x_df_trg: DataFrame, y_df_trg: DataFrame,
          method: Callable, dirname: str, n_folds: int, file: Any) -> list:
    cv_res = 0
    data, for_validation = [], []
    rng = default_rng()
    indices = rng.choice(1000, size=ITERS_NUM, replace=False)
    with Pool(multiprocessing.cpu_count() - 3) as cv_pool:  # буду считать на 5 ядрах, пока делаю дела параллельно
        # вот этот блок очень важно доделать
        for i in range(ITERS_NUM):
            folds_x_src, folds_y_src = split_folds(x_df_src, y_df_src, n_folds=n_folds)
            folds_x_trg, folds_y_trg = split_folds(x_df_trg, y_df_trg, n_folds=n_folds)
            fold_indices = list(range(n_folds))
            train_indices = np.random.choice(fold_indices, size=n_folds - 1, replace=False)
            valid_index = np.setdiff1d(fold_indices, train_indices)[0]
            src_valid = [folds_x_src[valid_index].to_numpy(), folds_y_src[valid_index].to_numpy().flatten()]
            trg_valid = [folds_x_trg[valid_index].to_numpy(), folds_y_trg[valid_index].to_numpy().flatten()]
            for_validation.append([src_valid, trg_valid])
            for df in [folds_x_trg, folds_y_trg, folds_y_src, folds_x_src]:
                df.pop(valid_index)
            src_train_x = pd.concat(folds_x_src)
            src_train_y = pd.concat(folds_y_src)
            trg_train_x = pd.concat(folds_x_trg)
            trg_train_y = pd.concat(folds_y_trg)
            data.append((src_train_x.to_numpy(), src_train_y.to_numpy().flatten(), trg_train_x.to_numpy(),
                         trg_train_y.to_numpy().flatten(), dirname, indices[i], False))
        data = cv_pool.starmap(method, data)

        # print temp results
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

        validation_scores = dict()
        valid_data_src, train_data_src = [], []
        valid_data_trg, train_data_trg = [], []
        for i, (model_res, validation_tuple) in enumerate(zip(data, for_validation)):
            model_src_train, model_trg_train = np.sqrt(model_res[1]), np.sqrt(model_res[2])
            tmp_fitness_function_src = SymbolicRegressionFitness(X_train=validation_tuple[0][0],
                                                                 y_train=validation_tuple[0][1])
            tmp_fitness_function_trg = SymbolicRegressionFitness(X_train=validation_tuple[1][0],
                                                                 y_train=validation_tuple[1][1])
            tmp_fitness_function_src.Evaluate(model_res[0])
            model_src_valid = np.sqrt(model_res[0].fitness)
            tmp_fitness_function_trg.Evaluate(model_res[0])
            model_trg_valid = np.sqrt(model_res[0].fitness)
            valid_data_trg.append(model_trg_valid)
            valid_data_src.append(model_src_valid)
            train_data_src.append(model_src_train)
            train_data_trg.append(model_trg_train)
            # Также сохраняем модель к результатам валидации
            validation_scores.update({"model{}".format(str(i + 1)): [model_data, model_src_train, model_src_valid,
                                                                     model_trg_train, model_trg_valid]})
        # строим гистограмму для source-data train-valid
        plot_cv_hist(data_train=train_data_src, data_valid=valid_data_src,
                     path_file="cv_res_source_data_63.jpg")
        # строим гистограмму для target-data train-valid
        plot_cv_hist(data_train=train_data_trg, data_valid=valid_data_trg,
                     path_file="cv_res_target_data_63.jpg")
        # здесь надо посчитать ошибки на target-train + target-validation
    return [data, validation_scores]
