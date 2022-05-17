from ITGP import ITGP
from dataset_parsing import initial_parse_data_and_save
from dataset_parsing import TEST_X_NAME, TEST_Y_NAME, TRAIN_X_NAME, TRAIN_Y_NAME, VALIDATION_X_NAME, VALIDATION_Y_NAME
from dataset_parsing import parse_x_y
import numpy as np
import pandas as pd

TRAIN_SIZE = 2500
TEST_SIZE = 500
VALIDATION_SIZE = 1000


def main():
    # for the same choice of target and c-resource datasets during training
    np.random.seed(1248)
    # The rest of the data from 4262, in add to 3000 already allocated, goes into testing
    # (verification of the model after the train and validation)
    # this line is run once to divide the data into those used to train the models
    # and those already used to check and evaluate the results
    # initial_parse_data_and_save()

    # тестовые значения, для отладки кода
    source_size = 1000
    target_size = 80

    x_train, y_train = pd.read_csv(TRAIN_X_NAME), pd.read_csv(TRAIN_Y_NAME)
    x_test, y_test = pd.read_csv(TEST_X_NAME), pd.read_csv(TEST_Y_NAME)
    y_test, y_train = np.array(y_test).flatten(), np.array(y_train).flatten()
    x_test, x_train = x_test.to_numpy(), x_train.to_numpy()
    source_data = parse_x_y(x_train, y_train, source_size)
    target_data = parse_x_y(x_test, y_test, target_size)
    res = ITGP(source_data[0], source_data[1], target_data[0], target_data[1], preload_models=True)


# if __name__ == 'main':
#     main()

main()
