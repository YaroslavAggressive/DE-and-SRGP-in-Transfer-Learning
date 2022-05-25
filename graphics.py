import matplotlib.pyplot as plt


def draw_hist(permutations: dict, name: str, norm: bool = False, save: bool = False, dirpath: str = ""):
    fig, ax = plt.subplots()
    ax.hist(list(permutations.keys()), permutations.values(), orientation='horizontal', density=norm)
    ax.grid()
    ax.set_xlabel("mean error after 100 permutations")
    ax.set_ylabel("column")
    ax.set_title(name)

    if save and len(dirpath) != 0:
        suffix = "permutations, " if not norm else "permutations, norm, "
        fig.savefig(dirpath + suffix + name)


def draw_temperature_plot(temperature_res: list, d_t: list, model_ind: int, name: str, save: bool = False,
                          dirpath: str = ""):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("d_T")
    ax.set_ylabel("Flowering time change")
    ax.plot(d_t, temperature_res)
    for t, t_val in zip(d_t, temperature_res):
        ax.scatter(t, t_val)
    ax.set_title("Time flowering, global warming of chickpea according to model #{}".format(model_ind))

    if save and len(dirpath) != 0:
        fig.savefig(dirpath + "temperature, " + name)
