import numpy as np
import os
from numpy.random import default_rng
from typing import Any
from pandas import DataFrame
import pandas as pd
from typing import Callable
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool

from simplegp.Nodes.BaseNode import Node
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from model_research import parse_data_per_iter, permutation_test
from model_research import global_warning_research, rain_research, sun_rad_research
from dataset_parsing import parse_x_y
from model_research import create_snp_columns
from ITGP import ITGP
from graphics import plot_cv_hist
from sympy import parse_expr, simplify, Symbol
from models_serialization import load_models
from ITGP import choose_best_model

DAYS_PREDICTION = 20  # number of days for prediction
SNIPS_RANGE = 6  # number of polymorphisms investigated in the framework of the work
ALLELES = ["AA", "AR", "RR"]  # number of alleles of each polymorphism
ITERS_NUM = 10  # for now 10 for ckecking how cross-validation process work and graphs buildings


def count_simple_functions(model: Node) -> dict:  # окей, это вроде бы сделано, но еше check
    result = dict()
    elementary_functions = ["log", "sin", "cos", "sqrt", "+", "-", " * ", "/", "**"]
    model_str = model.GetHumanExpression()
    for func in elementary_functions:
        tmp_count = model_str.count(func)
        result.update({func: tmp_count})
    return result


def replace_abstract_variables(model: Node, with_doy: bool = False) -> str:  # окей, это вроде бы сделано, но еше check
    model_str = model.GetHumanExpression()
    model_expr = parse_expr(model_str)
    # simplified_expr = simplify(model_expr)
    replacements = dict()
    if with_doy:  # crutch is needed because the sets of predictors, and therefore their order, are different
        replacements.update({"x0": "doy", "x1": "geo_id", "x20": "month", "x21": "year"})
    else:
        replacements.update({"x0": "geo_id"})

    weather_shift, snp_shift = 19, 1
    if with_doy:
        weather_shift += 3  # 'doy' goes before snp-s information, 'month' and 'year'  go after snp and before weather
        snp_shift += 1  # 'doy' goes before snp-s information, so shifting other columns

    snp_prefix = "snp"
    for i in range(SNIPS_RANGE):
        for j, allel in enumerate(ALLELES):
            column = snp_prefix + str(i + 1) + allel
            ind = j + snp_shift
            replacements.update({"x{}".format(len(ALLELES) * i + ind): column})

    climat_prefixes = ["tmin", "tmax", "srad", "rain", "dl"]

    for i in range(DAYS_PREDICTION):
        for j, prefix in enumerate(climat_prefixes):
            col_name = prefix + str(i)
            ind = weather_shift + len(climat_prefixes) * i + j
            replacements.update({"x{}".format(ind): col_name})

    # result = str(simplified_expr.subs(replacements))
    result = str(model_expr.subs(replacements))
    return result


def make_iter_dirs(res_path: str, iter_name: str) -> list:  # окей, это вроде бы сделано, но еше check
    existing_results = os.listdir(res_path)
    iter_path = res_path + "/" + iter_name
    functions_count_name = "top_func_counts"
    readable_expressions_name = "readable_funcs"
    permutations_name = "top_funcs_permutations"
    names = [permutations_name, functions_count_name, readable_expressions_name]

    if iter_name not in existing_results:
        os.mkdir(iter_path)

    iter_results = os.listdir(iter_path)
    res_paths = []
    for name in names:
        tmp_path = iter_path + "/" + name
        if name not in iter_results:
            os.mkdir(tmp_path)
        res_paths.append(tmp_path)
    return res_paths


def process_cv_results(iter_dir: str, iter_ind: int, size: int, seed: int, predictors: DataFrame, response: DataFrame,
                       with_doy: bool = False):
    res_path = "cv results"
    iter_name = "iter{}".format(iter_ind)
    permutations_dir, count_funcs_dir, readable_funcs_dir = make_iter_dirs(res_path=res_path, iter_name=iter_name)

    # process iter results and best model for each population
    parsed_data = parse_data_per_iter(x_df=predictors, y_df=response, seed=seed)
    iter_models = os.listdir(iter_dir)
    iter_models_serial = [iter_name for iter_name in iter_models if iter_name.startswith("models")]
    top_models = dict()
    for models_file in iter_models_serial:
        tmp_models = load_models(iter_dir + "/" + models_file, size)
        fitness = SymbolicRegressionFitness(X_train=predictors.to_numpy(),
                                            y_train=response.to_numpy().flatten())
        best_fit, best_model = np.inf, None
        for model in tmp_models:
            fitness.Evaluate(model)
            contestant = model.fitness
            if contestant <= best_fit:
                best_fit = contestant
                best_model = deepcopy(model)
        top_models.update({models_file: best_model})

    # count elementary functions in temp expression
    # for ind, top_model in top_models.items():
    #     file = open(count_funcs_dir + "/count_func_top_{}.txt".format(ind.replace(".txt", "")), "w")
    #     file.write("Top model from file {0}".format(ind.replace(".txt", "")) + "\n")
    #     functions_stat = count_simple_functions(top_model)
    #     for func, quan in functions_stat.items():
    #         file.write("Number of {0}: {1}".format(func, quan) + " \n")
    #     file.close()

    # # permutations tests
    # permutations_pool = Pool(mp.cpu_count() - 4)
    # data, files = [], []
    # for i, (top_model_ind, top_model) in enumerate(top_models.items()):
    #     data.append((seed, top_model, int(top_model_ind.replace("models", "").replace(".txt", "")), predictors,
    #                  response, False))
    #     file_name = "top_{}_permutations_errors.csv".format(top_model_ind.replace(".txt", ""))
    #     files.append(file_name)
    # perm_results = permutations_pool.starmap(permutation_test, data)
    # total_df = DataFrame()
    # for tmp_file, one_perm_res in zip(files, perm_results):
    #     permutations_df = DataFrame.from_dict(one_perm_res)
    #     total_df = pd.concat([total_df, permutations_df], axis=0)
    #     permutations_df.to_csv(permutations_dir + "/" + tmp_file, index=False, sep=";")
    # total_df.to_csv(permutations_df + "/all_cv_top_permutations.csv", index=False, sep=";")

    # make top expressions more simple and write in file for human reading
    expressions_file = "cv_results_over_last_cv_{}.txt".format(iter_ind)
    file = open(readable_funcs_dir + "/" + expressions_file, "w")
    for i, (top_model_ind, top_model) in enumerate(top_models.items()):
        true_expr = replace_abstract_variables(model=top_model, with_doy=with_doy)
        file.write("Model # {0}, from file {1}".format(str(i + 1), top_model_ind.replace(".txt", "")) + "\n")
        file.write("Expression: {}".format(true_expr) + "\n")
    file.write("Wrote every top model for iter {}".format(iter_ind) + "\n")
    file.close()

    # все что ниже можно доделать позже
    # warming tests

    # rain tests

    # sun rad tests


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
    with Pool(mp.cpu_count() - 3) as cv_pool:  # буду считать на 5 ядрах, пока делаю дела параллельно
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
            # res = ITGP(src_train_x.to_numpy(), src_train_y.to_numpy().flatten(), trg_train_x.to_numpy(),
            #            trg_train_y.to_numpy().flatten(), dirname, indices[i], False)
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
            # also save the model to the validation results
            validation_scores.update({"model{}".format(str(i + 1)): [model_data, model_src_train, model_src_valid,
                                                                     model_trg_train, model_trg_valid]})
        # building a histogram for source-data train-valid
        plot_cv_hist(data_train=train_data_src, data_valid=valid_data_src,
                     path_file="cv_res_source_data_63.jpg")
        # building a histogram for target-data train-valid
        plot_cv_hist(data_train=train_data_trg, data_valid=valid_data_trg,
                     path_file="cv_res_target_data_63.jpg")
        # here need to calculate the errors on target-train + target-validation
    return [data, validation_scores]
