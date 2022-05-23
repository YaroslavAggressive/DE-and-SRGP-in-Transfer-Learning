import matplotlib.pyplot as plt
import numpy as np


def draw_permutation_hist(permutations: dict, model_ind: int, name: str, save: bool = False, dirpath: str = ""):
    fig, ax = plt.subplots()
    ax.barh(list(permutations.keys()), permutations.values())
    ax.grid()
    ax.set_xlabel("mean error after 100 permutations")
    ax.set_ylabel("column")
    ax.set_title(name)
    # plt.show()

    if save and len(dirpath) != 0:
        fig.savefig(dirpath + "permutations, " + name)


def draw_temperature_plot(temperature_res: list, d_t: list, model_ind: int, name: str, save: bool = False, dirpath: str = ""):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("d_T")
    ax.set_ylabel("Flowering time change")
    ax.plot(d_t, temperature_res)
    ax.set_title("Time flowering in condition of global warming of chickpea according to model #{}".format(model_ind))
    # plt.show()

    if save and len(dirpath) != 0:
        fig.savefig(dirpath + "temperature, " + name)
