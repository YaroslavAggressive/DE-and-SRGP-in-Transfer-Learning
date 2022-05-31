import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def draw_hist(permutations: dict, name: str, norm: bool = False, save: bool = False, dirpath: str = ""):
    fig, ax = plt.subplots()
    if norm:  # в случае нормализованности гистограмма отрисовываем уровень в процентах от общего влияния столбца
        total = sum(permutations.values())
        for key, val in permutations.items():
            permutations[key] = val / total
    ax.barh(list(permutations.keys()), list(permutations.values()), orientation='horizontal')
    ax.grid()
    ax.set_xlabel("mean error after 100 permutations")
    ax.set_ylabel("column")
    ax.set_title(name)

    if save and len(dirpath) != 0:
        suffix = "permutations, " if not norm else "permutations, norm, "
        fig.savefig(dirpath + suffix + name)

    plt.close(fig)
    plt.clf()
    plt.cla()


def draw_temperature_plot(d_values: np.array, d_t: list, model_ind: int, snp_name: str, name: str, save: bool = False,
                          dir_path: str = ""):
    matplotlib.use('Agg')
    for j in range(d_values.shape[1]):
        fig, ax = plt.subplots()
        fig.set_size_inches(10., 10.)
        ax.grid()
        ax.set_xlabel("d_T")
        ax.set_ylabel("Flowering time change")
        ax.plot(d_t, d_values[:, j])
        for t, t_val in zip(d_t, d_values[:, j]):
            ax.scatter(t, t_val, c="b")
        ax.set_title("Time flowering of chickpea, climatic changes, {0}, model #{1}, flower #{2}"
                     .format(snp_name, model_ind, j))

        if save and len(dir_path) != 0:
            fig.savefig(dir_path + "temperature, " + name + ", flower #{}".format(j + 1) + ", " + snp_name)
        plt.close(fig)
        plt.clf()
        plt.cla()


def draw_srad_plot(d_values: np.array, d_rad: list, model_ind: int, snp_name: str, name: str, save: bool = False,
                   dir_path: str = ""):
    matplotlib.use('Agg')
    for j in range(d_values.shape[1]):
        fig, ax = plt.subplots()
        fig.set_size_inches(10., 10.)
        ax.grid()
        ax.set_xlabel("sun radiation change")  # in days - from 1 to 5
        ax.set_ylabel("Flowering time change")
        ax.plot(d_rad, d_values[:, j])
        for rad, rad_val in zip(d_rad, d_values[:, j]):
            ax.scatter(rad, rad_val, c="b")
        ax.set_title("Time flowering of chickpea, radiation changes, {0}, model #{1}, flower #{2}"
                     .format(snp_name, model_ind, j))
        if save and len(dir_path) != 0:
            fig.savefig(dir_path + "radiation, " + name + ", flower #{}".format(j + 1) + ", " + snp_name)
        plt.close(fig)
        plt.clf()
        plt.cla()


def draw_rain_plot(d_values: np.array, d_rad: list, model_ind: int, snp_name: str, name: str, save: bool = False,
                   dir_path: str = ""):
    matplotlib.use('Agg')
    for j in range(d_values.shape[1]):
        fig, ax = plt.subplots()
        fig.set_size_inches(10., 10.)
        ax.grid()
        ax.set_xlabel("rain level change")  # in days - from 1 to 5
        ax.set_ylabel("Flowering time change")
        ax.plot(d_rad, d_values[:, j])
        for rad, rad_val in zip(d_rad, d_values[:, j]):
            ax.scatter(rad, rad_val, c="b")
        ax.set_title("Time flowering of chickpea, rain level changes, {0}, model #{1}, flower #{2}"
                     .format(snp_name, model_ind, j))
        if save and len(dir_path) != 0:
            fig.savefig(dir_path + "rain, " + name + ", flower #{}".format(j + 1) + ", " + snp_name)
        plt.close(fig)
        plt.clf()
        plt.cla()


def plot_cv_hist(data_train: list, data_valid: list, path_file: str):
    mean_train = np.mean(data_train)
    mean_valid = np.mean(data_valid)
    fig, ax = plt.subplots()
    plt.xlabel("Error, days")
    plt.ylabel("Frequency")
    plt.grid()
    plt.hist(data_train, bins=np.arange(min(data_train), max(data_train) + 1), color="skyblue", label="train")
    plt.hist(data_valid, bins=np.arange(min(data_valid), max(data_valid) + 1), color="red", label="validation")
    plt.axvline(x=mean_valid, ymin=0, label="mean validation", c="b", linestyle='-')
    plt.axvline(x=mean_train, ymin=0, label="mean training", c="k", linestyle='-')
    plt.legend()
    plt.savefig(path_file)
    plt.close(fig)
    plt.clf()
    plt.cla()


def plot_response_change(x_true: list, x_prediction: list, path_file: str):
    true_min, true_max = min(x_true), max(x_true)
    fig, ax = plt.subplots()
    plt.title("Comparison between best model and ")
    plt.xlabel("Time flowering, days")
    plt.ylabel("Model prediction, days")
    for true_p, prediction_p in zip(x_true, x_prediction):
        plt.scatter(true_p, prediction_p, c="r")
    plt.plot([true_min, true_max], [true_min, true_max], 'bo', linestyle="--")
    plt.grid()
    plt.legend()
    plt.savefig(path_file)
    plt.close(fig)
    plt.clf()
    plt.cla()
