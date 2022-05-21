from simplegp.Nodes.BaseNode import Node
import pandas as pd
from pandas import DataFrame
import numpy as np
from copy import deepcopy
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
import re


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
