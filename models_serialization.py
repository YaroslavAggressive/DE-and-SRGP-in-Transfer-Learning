import numpy as np
import pickle
from population import Population

MODELS_SAVEFILE = "models_weights_info/models.txt"
WEIGHTS_SAVEFILE = "models_weights_info/weights.txt"

MODELS_FOR_CHECK = "models_weights_info/readable_models.txt"
WEIGHTS_FOR_CHECK = "models_weights_info/readable_weights.txt"


def save_models(filename: str, models: list):
    file = open(filename, "wb")
    for model in models:
        pickle.dump(model, file)
    file.close()


def save_weights(filename: str, weights_pop: Population):
    file = open(filename, "wb")
    pickle.dump(weights_pop, file)
    file.close()


def load_models(filename: str, n_models: int) -> list:
    models = []
    file = open(filename, "rb")
    for i in range(n_models):
        models.append(pickle.load(file))
    file.close()
    return models


def load_weights(filename: str) -> np.array:
    file = open(filename, "rb")
    pop_weights = pickle.load(file)
    file.close()
    return pop_weights


def readable_output_models(filename: str, models: list):
    with open(filename, "w") as file:
        for i, model in enumerate(models):
            file.write("Model # {}".format(i + 2) + "\n")
            file.write("Function: F(x) = {}".format(model.GetHumanExpression()) + "\n")
            file.write("Model fitness: {}".format(model.fitness) + "\n")
            file.write("" + "\n")
        file.close()


def readable_output_weights(filename: str, weights: list):
    with open(filename, "w") as file:
        for i, weight in enumerate(weights):
            file.write("Weights population #{}".format(i + 1) + "\n")
            file.write("Weight: {}".format(weight) + "\n")
        file.close()
