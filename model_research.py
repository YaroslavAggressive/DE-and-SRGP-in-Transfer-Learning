from simplegp.Nodes.BaseNode import Node
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pandas import DataFrame
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
import multiprocessing as mp
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
import os
from models_serialization import load_models, save_models, readable_output_models, read_top_models_size
from dataset_parsing import MERGED_DATASET, parse_autumn_spring, data_shuffle, parse_valid
from graphics import draw_hist, draw_temperature_plot

PERMUTATIONS_NUM = 100
# best models are stored here so that you don't have to search for them every time, but just explore
BEST_MODELS_FILE = "top_models/top_models_after_20_inters.txt"
BEST_MODELS_READABLE_FILE = "top_models/readable_top_models_after_20_inters.txt"
# here is just the size of the saved models, so that it is not serialized together with the models
BEST_MODELS_SIZE_FILE = "top_models/top_models_size.txt"
PERMUTATIONS_DIR = "permutations data"


def dataset_clustering(x_df_numpy: np.array, colnames: list, n_clusters: int = 5) -> list:
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(x_df_numpy)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(k_means.labels_):
        clusters[label].append(x_df_numpy[i])

    for j in range(n_clusters):
        clusters[j] = DataFrame(np.vstack(clusters[j]), columns=colnames)
    return clusters


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


def permutation_test(seed: int, model: Node, ind: int, x_df: DataFrame, y_df: DataFrame, save: bool = False) -> dict:
    # finding response dependence each predictor of input data
    parsed_data = parse_data_per_iter(x_df=x_df, y_df=y_df, seed=seed)
    np.random.seed(seed)
    fitness_function_src = SymbolicRegressionFitness(X_train=parsed_data[0].to_numpy(),
                                                     y_train=parsed_data[1].to_numpy().flatten())
    fitness_function_trg = SymbolicRegressionFitness(X_train=parsed_data[2].to_numpy(),
                                                     y_train=parsed_data[3].to_numpy().flatten())
    fitness_function_valid = SymbolicRegressionFitness(X_train=parsed_data[4].to_numpy(),
                                                       y_train=parsed_data[5].to_numpy().flatten())

    benchmark_error_src = get_fitness(fitness_function=fitness_function_src, model=model)
    benchmark_error_trg = get_fitness(fitness_function=fitness_function_trg, model=model)
    benchmark_error_valid = get_fitness(fitness_function=fitness_function_valid, model=model)
    column_error_change = dict()
    # доделать для 100 пересчеов (брать среднее значение)
    if save:
        name_dir = "test_{0}".format(ind)
        os.mkdir(PERMUTATIONS_DIR + "/" + name_dir)
    for i in range(PERMUTATIONS_NUM):
        x_df_copy = deepcopy(x_df)
        for column in x_df.columns:
            x_df_copy[column] = np.random.permutation(x_df_copy[column].values)
            if save:
                # saving dataset with EACH PERMUTATED COLUMN
                file_name = "perm_{0}_model_{1}_in_top.csv".format(i, ind)
                x_df_copy.to_csv(PERMUTATIONS_DIR + "/" + name_dir + "/" + file_name, index=False, sep=";")
        for column in x_df.columns:
            column_res = 0
            column_tmp = deepcopy(x_df[column])
            x_df[column] = x_df_copy[column]
            x_df_copy[column] = np.random.permutation(x_df_copy[column].values)
            parsed_data = parse_data_per_iter(x_df=x_df_copy, y_df=y_df, seed=seed)
            fitness_function_src.X_train = parsed_data[0].to_numpy()
            src_error = get_fitness(fitness_function=fitness_function_src, model=model)
            fitness_function_trg.X_train = parsed_data[2].to_numpy()
            trg_error = get_fitness(fitness_function=fitness_function_src, model=model)
            fitness_function_valid.X_train = parsed_data[4].to_numpy()
            valid_error = get_fitness(fitness_function=fitness_function_src, model=model)
            column_res += np.fabs(np.sqrt(src_error) + np.sqrt(trg_error) + np.sqrt(valid_error) -
                                  np.sqrt(benchmark_error_trg) - np.sqrt(benchmark_error_src) -
                                  np.sqrt(benchmark_error_valid))
            x_df[column] = column_tmp
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
    error = model_output - y_df.to_numpy().flatten()
    # теперь возвращаем не среднее,  а ошибку для каждого растения
    return error


def best_population_checking(x_df: DataFrame, y_df: DataFrame, iter_seeds: list, sizes_: dict):
    # for each case need to change seed name and dirname + 'models....txt' name
    size_ind = 17
    best_population = load_models("models_weights_info/iter17/models61.txt", sizes_["iter" + str(size_ind)])
    np.random.seed(iter_seeds[size_ind])
    x_src, y_src, x_trg, y_trg, x_valid, y_valid = parse_data_per_iter(x_df, y_df, iter_seeds[size_ind])
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


def test_model(model: Node, ind: int, x_df: DataFrame, y_df: DataFrame, seed: int, save: bool = False):
    parsed_data = parse_data_per_iter(x_df, y_df, seed)
    res_keys = ["src", "trg", "valid"]
    model_results = []
    for i, key in enumerate(res_keys):
        tmp_x, tmp_y = parsed_data[2 * i], parsed_data[2 * i + 1]
        temperature_results = []
        for d_t in [0.1 * i for i in range(20)]:
            tmp_res = global_warning_research(tmp_x, tmp_y, d_t, model)
            temperature_results.append(tmp_res)
        model_results.append([key, temperature_results])
    perm_res = permutation_test(seed, model, ind, x_df, y_df, save=True)
    model_results.append(perm_res)
    return model_results


def main_research(x_df: np.array, y_df: np.array, seeds_per_iter: dict, save: bool = False) -> list:
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
    print("Top models number : {}".format(len(top_models)))
    results = []
    titles = {"src": "Source data",
              "trg": "Target data",
              "valid": "Validation data"}
    research_pool = Pool(mp.cpu_count() - 3)
    data = []
    # clusters = dataset_clustering(x_df=x_df, y_df=y_df, n_clusters=18)  # число кластеров - по числу снипов AA/AR/RR
    for i, model_data in enumerate(top_models):
        data.append((model_data[0], i, deepcopy(x_df), deepcopy(y_df), seeds_per_iter[model_data[5]], save))
    calc_data = research_pool.starmap(test_model, data)
    for i, (model_res, model_data) in enumerate(zip(calc_data, top_models)):
        for dataset_res in model_res[0]:
            draw_temperature_plot(dataset_res[2], [0.1 * j for j in range(20)], i, titles[dataset_res[0]] +
                                  ", model # {}".format(str(i + 1)), True, "graphics/")
        draw_hist(model_res[1], "model # {}".format(str(i + 1)), norm=True, save=True, dirpath="graphics/")
        draw_hist(model_res[1], "model # {}".format(str(i + 1)), norm=False, save=True, dirpath="graphics/")
        results.append([model_data, model_res])

    return results
