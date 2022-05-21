from simplegp.Nodes.BaseNode import Node
import pandas as pd
from pandas import DataFrame
import numpy as np
from copy import deepcopy
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
import re
import os
from ITGP import choose_best_model
from models_serialization import load_models

NUMBER_OF_MODELS = 300


def permutation_test(model: Node, x_df: DataFrame, y_df: DataFrame) -> dict:
    # finding response dependence each predictor of input data
    fitness_function = SymbolicRegressionFitness(X_train=x_df, y_train=y_df)
    fitness_function.Evaluate(model)
    benchmark_error = model.fitness
    column_error_change = dict()
    for column in x_df.columns:
        x_df_copy = deepcopy(x_df)
        x_df_copy[column] = np.random.permutation(x_df_copy[column].values)
        fitness_function.X_train = x_df_copy
        fitness_function.Evaluate(model)
        column_error_change.update({column: np.fabs(model.fitness - benchmark_error)})
    return column_error_change


def emulate_global_warning(x_df: DataFrame, d_T: float, model: Node) -> DataFrame:
    # returns model prediction for time flowering of wild chickpea in changed climatic conditions
    temperature_prefixes = ["tmin", "tmax"]
    temperature_columns = []
    for column in x_df.columns:
        for prefix in temperature_prefixes:
            if re.search("^" + prefix, column) != 0:
                temperature_prefixes.append(column)
                break
    for column in temperature_columns:
        x_df[column] = x_df[column] + d_T

    return x_df


def best_model_from_iters(x_src: DataFrame, y_src: DataFrame, x_trg: DataFrame, y_trg: DataFrame, x_valid: DataFrame,
                          y_valid: DataFrame, files: list):
    best_model = None
    src_fitness = SymbolicRegressionFitness(X_train=x_src.to_numpy(), y_train=y_src.to_numpy().flatten())
    trg_fitness = SymbolicRegressionFitness(X_train=x_trg.to_numpy(), y_train=y_trg.to_numpy().flatten())
    top_models_data = []
    for file in files:
        models = load_models(file, NUMBER_OF_MODELS)
        for model in models:
            trg_fitness.Evaluate(model)
        best_model_data = choose_best_model(models, src_fitness)
        top_models_data.append(best_model_data)

    valid_fitness = SymbolicRegressionFitness(X_train=x_valid.to_numpy(), y_train=y_valid.to_numpy().flatten())
    best_valid_fit, best_trg_fit, best_src_fit = np.inf, np.inf, np.inf
    for model_data in top_models_data:
        valid_fitness.Evaluate(model_data[0])
        model_fitness_valid = model_data[0].fitness
        if best_src_fit >= model_data[1] and best_valid_fit + best_src_fit + best_trg_fit >= \
            model_fitness_valid + model_data[1] + model_data[2]:
            best_src_fit, best_trg_fit, best_valid_fit = model_data[1], model_data[2], model_fitness_valid
            best_model = deepcopy(model_data[0])
    x_df, y_df = pd.concat([x_src, x_trg, x_valid]), pd.concat([y_src, y_trg, y_valid])
    permutations_res = permutation_test(x_df, y_df, best_model)
    climat_test = emulate_global_warning(x_df, 0.5, best_model)


dirpath = "models_weights_info"
dirs = os.listdir(dirpath)
files = []
for directory in dirs:
    dir_files = os.listdir(dirpath + "/" + directory)
    for file in dir_files:
        if re.search("^models", file) != 0:


