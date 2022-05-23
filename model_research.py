from simplegp.Nodes.BaseNode import Node
import pandas as pd
from pandas import DataFrame
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
import multiprocessing as mp
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
import os
from models_serialization import load_models, save_models, readable_output_models
from dataset_parsing import get_data_response, MERGED_DATASET, parse_autumn_spring, data_shuffle, parse_valid
from graphics import draw_permutation_hist, draw_temperature_plot

PERMUTATIONS_NUM = 100
BEST_MODELS_FILE = "top_models/top_models_after_20_inters.txt"  # здесь хранятся лучшие модели, чтобы каждый раз их не искать, а просто исследовать
BEST_MODELS_READABLE_FILE = "top_models/readable_top_models_after_20_inters.txt"
BEST_MODELS_SIZE_FILE = "top_models/top_models_size.txt"  # здесь просто лежит размер сохраненных моделей, чтобы его не сериализовывать вместе с моделями


def read_top_models_size(filename: str) -> int:
    file_size = open(filename, "r")
    text = file_size.readlines()
    if len(text) != 1:
        raise Exception("Invalid size of file with models population size")
    else:
        try:
            size = int(text[0])
            return size
        except Exception:
            raise Exception("Can't read size of top models population from chosen file, please, check it before using")


def find_save_best_models(x_df: np.array, y_df: np.array, all_iters: dict, sizes_per_iter: dict, seeds_per_iter: dict):
    top_models = []
    for i, directory in enumerate(all_iters.keys()):
        iter_files = all_iters[directory]
        seed = seeds_per_iter[directory]
        x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(x_df, y_df, seed)
        fitness_function_src = SymbolicRegressionFitness(X_train=x_src.to_numpy(), y_train=y_src.to_numpy().flatten())
        fitness_function_trg = SymbolicRegressionFitness(X_train=x_trg.to_numpy(), y_train=y_trg.to_numpy().flatten())
        fitness_function_valid = SymbolicRegressionFitness(X_train=x_valid.to_numpy(),
                                                           y_train=y_valid.to_numpy().flatten())
        for file_ in iter_files:
            models = load_models(file_, sizes_per_iter[directory])
            for model in models:
                fit_src = get_fitness(fitness_function_src, model)
                fit_trg = get_fitness(fitness_function_trg, model)
                fit_valid = get_fitness(fitness_function_valid, model)
                if fit_valid <= 22 and fit_src <= 12 and fit_trg <= 14:
                    top_models.append([deepcopy(model), fit_src, fit_trg, fit_valid, file_, directory])
    save_models(BEST_MODELS_FILE, top_models)
    readable_output_models(BEST_MODELS_READABLE_FILE, [model[0] for model in top_models],
                           SymbolicRegressionFitness(X_train=x_df.to_numpy(), y_train=y_df.to_numpy().flatten()))
    file_size = open(BEST_MODELS_SIZE_FILE, "w")
    file_size.write(str(len(top_models)))
    file_size.close()


def permutation_test(model: Node, x_df: DataFrame, y_df: DataFrame) -> dict:
    # finding response dependence each predictor of input data
    fitness_function = SymbolicRegressionFitness(X_train=x_df.to_numpy(), y_train=y_df.to_numpy().flatten())
    fitness_function.Evaluate(model)
    benchmark_error = model.fitness
    column_error_change = dict()
    # доделать для 100 пересчеов (брать среднее значение)
    for column in x_df.columns:
        column_res = 0
        for i in range(PERMUTATIONS_NUM):
            x_df_copy = deepcopy(x_df)
            x_df_copy[column] = np.random.permutation(x_df_copy[column].values)
            fitness_function.X_train = x_df_copy.to_numpy()
            fitness_function.Evaluate(model)
            column_res += np.fabs(np.sqrt(model.fitness) - np.sqrt(benchmark_error))
        if column_res > 0.:
            column_error_change.update({column: column_res / PERMUTATIONS_NUM})
    return column_error_change


def global_warning_research(x_df: DataFrame, y_df: DataFrame, d_T: float, model: Node) -> np.array:
    # returns model prediction for time flowering of wild chickpea in changed climatic conditions
    temperature_prefixes = ["tmin", "tmax"]
    temperature_columns = []
    for column in x_df.columns:
        for prefix in temperature_prefixes:
            if column.startswith(prefix):
                temperature_columns.append(column)
                break
    x_df_copy = deepcopy(x_df)
    for column in temperature_columns:
        x_df_copy[column] = x_df[column] + d_T
    # temperature data has been changed, now starting research with model

    model_output = model.GetOutput(x_df_copy.to_numpy())
    error = np.fabs(model_output - y_df.to_numpy().flatten())
    return np.mean(error)


def best_population_checking(x_df: DataFrame, y_df: DataFrame, iter_seeds: list, sizes_: dict):
    # for each case need to change seed name and dirname + 'models....txt' name
    size_ind = 17
    best_population = load_models("models_weights_info/iter17/models61.txt", sizes_["iter" + str(size_ind)])
    np.random.seed(iter_seeds[size_ind])
    x_src, y_src, x_trg, y_trg = parse_autumn_spring(x_df, y_df)
    x_src, y_src = data_shuffle(x_src, y_src)
    x_trg, y_trg = data_shuffle(x_trg, y_trg)
    src_size = x_src.shape[0]
    trg_size = src_size // 5
    valid_size = x_trg.shape[0] - trg_size
    x_valid, y_valid = parse_valid(x_trg, y_trg, valid_size)
    x_numpy, y_numpy = x_valid.to_numpy(), y_valid.to_numpy().flatten()
    train_function = SymbolicRegressionFitness(X_train=x_numpy, y_train=y_numpy)
    for i, model in enumerate(best_population):
        train_function.Evaluate(model)
        print("Model # {}".format(i + 2))
        print("Fitness on validation data: {}".format(np.sqrt(model.fitness)))


def parse_data_per_iter(x_df: DataFrame, y_df: DataFrame, seed: int) -> list:
    np.random.seed(seed)
    x_src, y_src, x_trg, y_trg = parse_autumn_spring(x_df, y_df)
    x_src, y_src = data_shuffle(x_src, y_src)
    x_trg, y_trg = data_shuffle(x_trg, y_trg)
    src_size = x_src.shape[0]
    trg_size = src_size // 5
    valid_size = x_trg.shape[0] - trg_size
    x_valid, y_valid = parse_valid(x_trg, y_trg, valid_size)
    return [x_src, y_src, x_trg, y_trg, x_valid, y_valid]


def get_fitness(fitness_function: SymbolicRegressionFitness, model: Node) -> float:
    fitness_function.Evaluate(model)
    return np.sqrt(model.fitness)


def test_iter(x_df: DataFrame, y_df: DataFrame, seed: int, pop_size: int, iter_name: str) -> list:
    x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(x_df, y_df, seed)
    iter_files = os.listdir("models_weights_info/" + iter_name)
    iter_models = []
    for file_ in iter_files:
        if file_.startswith("models"):
            iter_models.append("models_weights_info/" + iter_name + "/" + file_)
    fitness_function_valid = SymbolicRegressionFitness(X_train=x_src.to_numpy(), y_train=y_src.to_numpy().flatten())
    fitness_function_src = SymbolicRegressionFitness(X_train=x_trg.to_numpy(), y_train=y_trg.to_numpy().flatten())
    fitness_function_trg = SymbolicRegressionFitness(X_train=x_valid.to_numpy(), y_train=y_valid.to_numpy().flatten())
    top_models = []
    for model_file in iter_models:
        tmp_models = load_models(model_file, pop_size)
        for model in tmp_models:
            src_fit = get_fitness(fitness_function_src, model)
            trg_fit = get_fitness(fitness_function_trg, model)
            valid_fit = get_fitness(fitness_function_valid, model)
            if src_fit <= 10 and trg_fit <= 10 and valid_fit <= 16:
                top_models.append([deepcopy(model), iter_name, model_file, seed])
    return top_models


def test_model(model: Node, x_df: DataFrame, y_df: DataFrame, seed: int):
    parsed_data = parse_data_per_iter(x_df, y_df, seed)
    res_keys = ["src", "trg", "valid"]
    model_results = []
    for i, key in enumerate(res_keys):
        tmp_x, tmp_y = parsed_data[2 * i], parsed_data[2 * i + 1]
        perm_res = permutation_test(model, tmp_x, tmp_y)
        temperature_results = []
        for d_t in [0.1 * i for i in range(20)]:
            tmp_res = global_warning_research(tmp_x, tmp_y, d_t, model)
            temperature_results.append(tmp_res)
        model_results.append([key, perm_res, temperature_results])
    return model_results


def tyuki_criterion():
    pass


def main_research(x_df: np.array, y_df: np.array, all_iters: dict, sizes_per_iter: dict, seeds_per_iter: dict) -> list:
    models_size = read_top_models_size(BEST_MODELS_SIZE_FILE)
    top_models = load_models(BEST_MODELS_FILE, models_size)

    best_model_data = top_models[0]
    for model_data in top_models:
        if sum(model_data[1:4]) <= sum(best_model_data[1:4]):
            best_model_data = deepcopy(best_model_data)
    print("Best model: {}".format(best_model_data[0].GetHumanExpression()))
    print("Found in" + best_model_data[5])
    print("Saved in file '{}'".format(best_model_data[4]))
    print("Model total fitness: {0} (source) + {1} (target) + {2} (validation)".format(*best_model_data[1:4]))
    print("###########################")
    print(len(top_models))
    results = []
    titles = {"src": "Source data",
              "trg": "Target data",
              "valid": "Validation data"}
    research_pool = Pool(mp.cpu_count() - 3)
    data = []
    for i, model_data in enumerate(top_models):
        data.append((model_data[0], deepcopy(x_df), deepcopy(y_df), seeds_per_iter[model_data[5]]))
    calc_data = research_pool.starmap(test_model, data)
    for i, (model_res, model_data) in enumerate(zip(calc_data, top_models)):
        for dataset_res in model_res:
            draw_temperature_plot(dataset_res[2], [0.1 * j for j in range(20)], i, titles[dataset_res[0]] +
                                  ", model # {}".format(str(i + 1)), True, "graphics/")
            draw_permutation_hist(dataset_res[1], i, titles[dataset_res[0]] + ", model # {}".format(str(i + 1)), True,
                                  "graphics/")
        results.append([model_data, model_res])

    # for i, model_data in enumerate(top_models):
    #     model_res = test_model(model=model_data[0], x_df=x_df, y_df=y_df, seed=seeds_per_iter[model_data[5]])
    #     for dataset_res in model_res:
    #         draw_temperature_plot(dataset_res[2], [0.1 * j for j in range(20)], i, titles[dataset_res[0]] +
    #                               ", model # {}".format(str(i + 1)), True, "graphics/")
    #         draw_permutation_hist(dataset_res[1], i, titles[dataset_res[0]] + ", model # {}".format(str(i + 1)), True,
    #                               "graphics/")
    #     results.append([model_data, model_res])

    return results


def draw_results(results: list):
    titles = {"src": "Source data",
              "trg": "Target data",
              "valid": "Validation data"}
    for i, tmp_data in enumerate(results):
        model_data, model_res = tmp_data[0], tmp_data[1]
        for dataset_res in model_res:
            draw_temperature_plot(dataset_res[2], [0.1 * i for i in range(20)], i, titles[dataset_res[0]] +
                                  ", model # {}".format(i + 1), True, "graphics/")
            draw_permutation_hist(dataset_res[1], i, titles[dataset_res[0]] + ", model # {}".format(i + 1), True,
                                  "graphics/")


# # test seeds and models populations sizes
# mp.freeze_support()
# seeds = [3, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
# pop_sizes = [300, 300, 400, 300, 300, 300, 300, 300, 300, 300, 300, 300, 200, 300, 300, 300, 300, 300, 300, 300]
# pop_sizes_dict = {"iter" + str(i): pop_sizes[i] for i in range(len(pop_sizes))}
# seeds_dict = {"iter" + str(i): seeds[i] for i in range(len(seeds))}
#
# # total dataset and response reading
# redundant_column_name = "Unnamed: 0"
# response = get_data_response()  # parameter for prediction in future
# predictors = pd.read_csv(MERGED_DATASET, sep=";")
# del predictors[redundant_column_name]
#
# # making main research all over the fiven data
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
# #                       sizes_per_iter=pop_sizes_dict, seeds_per_iter=seeds_dict)
#
# # process_data = best_model_over_research(predictors.to_numpy(), response.to_numpy().flatten(), files_models,
# #                                         pop_sizes_dict)
# # best_population_checking(predictors, response, iter_seeds=seeds, sizes_=pop_sizes_dict)
#
# research = main_research(x_df=predictors, y_df=response, all_iters=files_models,
#                          sizes_per_iter=pop_sizes_dict, seeds_per_iter=seeds_dict)
