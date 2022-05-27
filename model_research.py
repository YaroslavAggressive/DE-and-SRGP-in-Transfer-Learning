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
from dataset_parsing import MERGED_DATASET, parse_autumn_spring, data_shuffle, parse_valid, create_snp_columns
from graphics import draw_hist, draw_temperature_plot, draw_srad_plot, draw_rain_plot

PERMUTATIONS_NUM = 100
# best models are stored here so that you don't have to search for them every time, but just explore
BEST_MODELS_FILE = "top_models_after_iter"
BEST_MODELS_READABLE_FILE = "readable_top_models_after_iter"
SUFFIX = ".txt"
# here is just the size of the saved models, so that it is not serialized together with the models
TOP_MODELS_DIR = "top_models"
BEST_MODELS_SIZE_FILE = "top_models_size"
PERMUTATIONS_DIR = "permutations data"
RAIN_MODES = {"full": 0.0, "half": 0.5}  # доля снижения осадков
RAD_MODES = {"half": 0.5, "quarter": 0.25, "tenth": 0.1}


def find_save_best_models(x_df: np.array, y_df: np.array, all_iters: dict, sizes_per_iter: dict, seeds_per_iter: dict,
                          iters_left: int, iters_right: int):
    top_models = []
    for i, directory in enumerate(all_iters.keys()):
        iter_ind = directory.replace("iter", "")
        iter_ind = int(iter_ind)
        if not iters_left <= iter_ind <= iters_right:
            continue
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
    top_models_path = TOP_MODELS_DIR + "/" + BEST_MODELS_FILE + "_" + str(iters_left) + "_" + str(iters_right)
    if not os.path.exists(top_models_path):
        os.mkdir(top_models_path)
    models_filename = BEST_MODELS_FILE + "_" + str(iters_left) + "_" + str(iters_right) + SUFFIX
    readable_models_filename = BEST_MODELS_READABLE_FILE + "_" + str(iters_left) + "_" + str(iters_right) + SUFFIX
    models_size_filename = BEST_MODELS_SIZE_FILE + "_" + str(iters_left) + "_" + str(iters_right) + SUFFIX
    save_models(top_models_path + "/" + models_filename, top_models)
    readable_output_models(top_models_path + "/" + readable_models_filename, [model[0] for model in top_models],
                           SymbolicRegressionFitness(X_train=x_df.to_numpy(), y_train=y_df.to_numpy().flatten()))
    file_size = open(top_models_path + "/" + models_size_filename, "w")
    file_size.write(str(len(top_models)))
    file_size.close()


def permutation_test(seed: int, model: Node, ind: int, x_df: DataFrame, y_df: DataFrame, save: bool = False) -> dict:
    # finding response dependence each predictor of input data
    np.random.seed(seed)
    fitness_function = SymbolicRegressionFitness(X_train=x_df.to_numpy(), y_train=y_df.to_numpy().flatten())
    benchmark_error = get_fitness(fitness_function=fitness_function, model=model)
    column_error_change = dict()

    # if save:
    #     name_dir = "test_{0}".format(ind)
    #     os.mkdir(PERMUTATIONS_DIR + "/" + name_dir)
    for i in range(PERMUTATIONS_NUM):
        x_df_copy = deepcopy(x_df)
        for column in x_df.columns:
            x_df_copy[column] = np.random.permutation(x_df_copy[column].values)
            # if save:
            #     # saving dataset with EACH PERMUTATED COLUMN
            #     file_name = "perm_{0}_model_{1}_in_top.csv".format(i, ind)
            #     x_df_copy.to_csv(PERMUTATIONS_DIR + "/" + name_dir + "/" + file_name, index=False, sep=";")
        for column in x_df.columns:
            column_res = 0 if column not in column_error_change.keys() else column_error_change[column]
            column_tmp = deepcopy(x_df[column])
            x_df[column] = x_df_copy[column]
            fitness_function.X_train = x_df.to_numpy()
            error = get_fitness(fitness_function=fitness_function, model=model)
            column_res += np.fabs(error - benchmark_error)
            x_df[column] = column_tmp
            if column_res > 1e-7:
                column_error_change.update({column: column_res})
    for column in x_df.columns:
        if column in column_error_change.keys():
            val = column_error_change[column]
            column_error_change.update({column: val / PERMUTATIONS_NUM})
    return column_error_change


def get_prefix_column(df_columns: list, prefixes: list) -> list:
    res_columns = []
    for column in df_columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                res_columns.append(column)
                break
    return res_columns


def global_warning_research(x_df: DataFrame, y_df: DataFrame, d_T: float, model: Node, only_min: bool = False) -> \
    np.array:
    # returns model prediction for time flowering of wild chickpea in changed climatic conditions
    temperature_prefixes = ["tmin", "tmax"] if not only_min else ["tmin"]
    temperature_columns = get_prefix_column(df_columns=list(x_df.columns), prefixes=temperature_prefixes)
    x_df_copy = deepcopy(x_df)
    for column in temperature_columns:
        x_df_copy[column] = x_df[column] + d_T
    # temperature data has been changed, now starting research with model

    model_output = model.GetOutput(x_df_copy.to_numpy())
    change = model_output - y_df.to_numpy().flatten()
    return change


def sun_rad_research(x_df: DataFrame, y_df: DataFrame, model: Node, mode: str = "half", days_borders: list = None) -> np.array:
    if days_borders[0] < 0 or days_borders[1] > 19:
        raise ValueError("Incorrect days indexing, please check your input")
    sun_rad_prefixes = ["srad"]
    rad_mode = RAD_MODES[mode]
    x_df_copy = deepcopy(x_df)
    for i in range(days_borders[0], days_borders[1] + 1):
        col_name = sun_rad_prefixes[0] + str(i)
        x_df_copy[col_name] = x_df_copy[col_name] * rad_mode
    model_output = model.GetOutput(x_df_copy.to_numpy())
    change = model_output - y_df.to_numpy().flatten()
    return change


def rain_research(x_df: DataFrame, y_df: DataFrame, model: Node, mode: str = "full", days_borders: list = None) -> np.array:
    if days_borders[0] < 0 or days_borders[1] > 19:
        raise ValueError("Incorrect days indexing, please check your input")
    rain_prefix = ["rain"]
    change_coeff = RAIN_MODES[mode]
    x_df_copy = deepcopy(x_df)
    for i in range(days_borders[0], days_borders[1] + 1):
        col_name = rain_prefix[0] + str(i)
        x_df_copy[col_name] = x_df_copy[col_name] * change_coeff
    model_output = model.GetOutput(x_df_copy.to_numpy())
    change = model_output - y_df.to_numpy().flatten()
    return change


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
    x_trg_index, x_valid_index = x_trg.index, x_valid.index
    trg_not_in_valid = np.setdiff1d(x_trg_index, x_valid_index)
    return [x_src, y_src, x_trg.loc[trg_not_in_valid], y_trg.loc[trg_not_in_valid], x_valid, y_valid]


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


def test_model(model: Node, ind: int, x_df: DataFrame, y_df: DataFrame, seed: int, save: bool = False, param: str = ""):
    parsed_data = parse_data_per_iter(x_df, y_df, seed)
    res_keys = ["src", "trg", "valid"]
    if param == "temp":
        model_results = []
        for i, key in enumerate(res_keys):
            tmp_x, tmp_y = parsed_data[2 * i], parsed_data[2 * i + 1]
            temperature_results = []
            for d_t in [0.1 * i for i in range(20)]:
                temperature_results.append(global_warning_research(tmp_x, tmp_y, d_t, model))
            temperature_results = np.vstack(temperature_results)
            model_results.append([key, temperature_results, tmp_x.index])
        return model_results
    elif param == "perm":
        perm_res = permutation_test(seed, model, ind, x_df, y_df, save=True)
        return perm_res
    elif param == "srad":
        model_results = []
        for i, key in enumerate(res_keys):
            tmp_x, tmp_y = parsed_data[2 * i], parsed_data[2 * i + 1]
            sun_rad_results = []
            for days in range(1, 5):
                sun_rad_results.append(sun_rad_research(tmp_x, tmp_y, model=model, mode="half", days_borders=[0, days]))
            sun_rad_results = np.vstack(sun_rad_results)
            model_results.append([key, sun_rad_results, tmp_x.index])
        return model_results
    elif param == "rain":
        model_results = []
        for i, key in enumerate(res_keys):
            tmp_x, tmp_y = parsed_data[2 * i], parsed_data[2 * i + 1]
            sun_rad_results = []
            for days in range(1, 5):
                sun_rad_results.append(rain_research(tmp_x, tmp_y, model=model, mode="full", days_borders=[0, days]))
            sun_rad_results = np.vstack(sun_rad_results)
            model_results.append([key, sun_rad_results, tmp_x.index])
        return model_results


def main_research(x_df: np.array, y_df: np.array, seeds_per_iter: dict, save: bool = False, iter_left: int = 0,
                  iter_right: int = 0) -> list:
    models_size = read_top_models_size(TOP_MODELS_DIR + "/" + BEST_MODELS_SIZE_FILE + "_" + str(iter_left) + "_" +
                                       str(iter_right) + SUFFIX)
    top_models = load_models(TOP_MODELS_DIR + "/" + BEST_MODELS_FILE + "_" + str(iter_left) + "_" + str(iter_right)
                             + SUFFIX, models_size)

    best_model_data = top_models[0]
    for model_data in top_models:
        if sum(model_data[1:4]) <= sum(best_model_data[1:4]):
            best_model_data = deepcopy(best_model_data)
    print("Best model: {}".format(best_model_data[0].GetHumanExpression()))
    print("Found in " + best_model_data[5])
    print("Saved in file '{}'".format(best_model_data[4]))
    print("Model total fitness: {0} (source) + {1} (target) + {2} (validation)".format(*best_model_data[1:4]))
    print("###########################")
    print("Top models number : {}".format(len(top_models)))

    # блок анализа значимости каждого фактора в предскзаании времени цветения образцов нута лучшими отобранными моделями
    results = []
    research_pool = Pool(mp.cpu_count() - 2)
    data = []
    # for i, model_data in enumerate(top_models):
    #     model_res = test_model(model_data[0], i, deepcopy(x_df), deepcopy(y_df), seeds_per_iter[model_data[5]], save, "perm")
    #     draw_hist(model_res, "model # {}".format(str(i + 1)), norm=False, save=True, dirpath="graphics/")
    #     draw_hist(model_res, "model # {}".format(str(i + 1)), norm=True, save=True, dirpath="graphics/")
    #     data.append(model_res)
    for i, model_data in enumerate(top_models):
        data.append((model_data[0], i, deepcopy(x_df), deepcopy(y_df), seeds_per_iter[model_data[5]], save, "perm"))
    calc_data = research_pool.starmap(test_model, data)
    for i, (model_res, model_data) in enumerate(zip(calc_data, top_models)):
        draw_hist(model_res, "model # {}".format(str(i + 1)), norm=False, save=True, dirpath="graphics/")
        draw_hist(model_res, "model # {}".format(str(i + 1)), norm=True, save=True, dirpath="graphics/")
        results.append([model_data, model_res])

    # блок построения графиков изменения предсказания времени цветения для каждого растения в датасете
    # в условиях изменения (в данном случае повышения) максимальной и минимальной дневной температуры воздуха
    # titles = {"src": "Source data",
    #           "trg": "Target data",
    #           "valid": "Validation data"}
    # for j, model_data in enumerate(top_models):
    #     model_path = "graphics/model {}".format(j)
    #     if not os.path.exists(model_path):
    #         os.mkdir(path=model_path)
    #     for i in range(6):  # проходим по всем снипам нута
    #         snp_columns_i = create_snp_columns(i + 1)
    #         x_snp_data = x_df[(x_df[snp_columns_i[0]] == 1.) |
    #                           (x_df[snp_columns_i[1]] == 1.) |
    #                           (x_df[snp_columns_i[2]] == 1.)]
    #         y_snp_data = y_df.loc[x_snp_data.index]
    #         tmp_temperature_test = test_model(model_data[0], i, x_snp_data, y_snp_data,
    #                                           seeds_per_iter[model_data[5]], save, "temp")
    #         for dataset_res in tmp_temperature_test:
    #             errors_data = dataset_res[1]
    #             draw_temperature_plot(errors_data, [0.1 * j for j in range(20)], i, "snp{}".format(i),
    #                                   titles[dataset_res[0]] + ", model # {}".format(str(i + 1)), True,
    #                                   "graphics/model {}".format(j) + "/")

    # блок изучения влияние изменения солнечной радиации на отклик модели
    # titles = {"src": "Source data",
    #           "trg": "Target data",
    #           "valid": "Validation data"}
    # for j, model_data in enumerate(top_models):
    #     model_path = "graphics/sun_rad/model {}".format(j)
    #     if not os.path.exists(model_path):
    #         os.mkdir(path=model_path)
    #     for i in range(6):  # проходим по всем снипам нута
    #         snp_columns_i = create_snp_columns(i + 1)
    #         x_snp_data = x_df[(x_df[snp_columns_i[0]] == 1.) |
    #                           (x_df[snp_columns_i[1]] == 1.) |
    #                           (x_df[snp_columns_i[2]] == 1.)]
    #         y_snp_data = y_df.loc[x_snp_data.index]
    #         tmp_rad_test = test_model(model_data[0], i, x_snp_data, y_snp_data,
    #                                   seeds_per_iter[model_data[5]], save, "srad")
    #         for dataset_res in tmp_rad_test:
    #             errors_data = dataset_res[1]
    #             draw_srad_plot(errors_data, list(range(2, 6)), i, "snp{}".format(i),
    #                            titles[dataset_res[0]] + ", model # {}".format(str(i + 1)), True, model_path + "/")

    # блок изучения влияние изменения уровня осадков на отклик модели
    titles = {"src": "Source data",
              "trg": "Target data",
              "valid": "Validation data"}
    for j, model_data in enumerate(top_models):
        model_path = "graphics/rain/model {}".format(j)
        if not os.path.exists(model_path):
            os.mkdir(path=model_path)
        for i in range(6):  # проходим по всем снипам нута
            snp_columns_i = create_snp_columns(i + 1)
            x_snp_data = x_df[(x_df[snp_columns_i[0]] == 1.) |
                              (x_df[snp_columns_i[1]] == 1.) |
                              (x_df[snp_columns_i[2]] == 1.)]
            y_snp_data = y_df.loc[x_snp_data.index]
            tmp_rain_test = test_model(model_data[0], i, x_snp_data, y_snp_data,
                                       seeds_per_iter[model_data[5]], save, "rain")
            for dataset_res in tmp_rain_test:
                errors_data = dataset_res[1]
                draw_rain_plot(errors_data, list(range(2, 6)), i, "snp{}".format(i),
                               titles[dataset_res[0]] + ", model # {}".format(str(i + 1)), True, model_path + "/")

    return results
