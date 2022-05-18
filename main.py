from ITGP import ITGP
from dataset_parsing import initial_parse_data_and_save, MERGED_DATASET
from dataset_parsing import parse_x_y, parse_per_key, parse_per_ageev_state, parse_per_season, parse_per_snp
from dataset_parsing import SEASON_KEY, GEO_ID_KEY, DATASET_SEASONS, SNP_KEYS, get_data_response
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA

TRAIN_SIZE = 2500
TEST_SIZE = 500
VALIDATION_SIZE = 1000
DAYS_PER_SNIP = 20  # число дней для предсказания погоды


def main():
    # for the same choice of target and c-resource datasets during training
    time = datetime.now()
    np.random.seed(time.microsecond)

    # тестовые значения, для отладки кода
    source_size = 1000
    target_size = 100
    merged_dataset = pd.read_csv(MERGED_DATASET, sep=';')
    y_merged = get_data_response()
    redundant_column_name = "Unnamed: 0"
    del merged_dataset[redundant_column_name]
    for i in range(6):
        snp_ind = 3
        dataset_data = parse_per_snp(merged_dataset, y_merged, snp_ind)
        dataset_data[0].to_csv("snp_datasets/data_no_snp_" + str(snp_ind))
        dataset_data[1].to_csv("snp_datasets/data_no_snp_" + str(snp_ind) + "response")

    # x_train, y_train = pd.read_csv(TRAIN_X_NAME), pd.read_csv(TRAIN_Y_NAME)
    # x_test, y_test = pd.read_csv(TEST_X_NAME), pd.read_csv(TEST_Y_NAME)
    # y_test, y_train = np.array(y_test).flatten(), np.array(y_train).flatten()
    # x_test, x_train = x_test.to_numpy(), x_train.to_numpy()
    # np.random.seed(1)
    # source_data = parse_x_y(dataset_data[0].to_numpy(), dataset_data[1].to_numpy().flatten(), source_size)
    # target_data = parse_x_y(dataset_data[2].to_numpy(), dataset_data[3].to_numpy().flatten(), target_size)
    # res = ITGP(source_data[0], source_data[1], target_data[0], target_data[1], preload_models=False)


# if __name__ == 'main':
#     main()

main()
