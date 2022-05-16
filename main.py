from ITGP import ITGP
from models_serialization import load_models, load_weights, save_models, save_weights, MODELS_SAVEFILE, WEIGHTS_SAVEFILE
from models_serialization import readable_output_weights, readable_output_models, MODELS_FOR_CHECK, WEIGHTS_FOR_CHECK
from dataset_parsing import MERGED_DATASET
from dataset_parsing import get_meteo_data, get_ssm_data, get_data_response, parse_source_target, merge_data
import numpy as np
import pandas as pd


def main():
    # for the same choice of target and c-resource datasets during training
    np.random.seed(1)

    # merging meteo and ssm data to get completed data
    # get prediction parameters, merging them with permutations and additions and finally getting data response
    df_ssm = get_ssm_data()
    df_meteo = get_meteo_data()
    y_nut = get_data_response()  # parameter for prediction in future
    # IF DATA HAVEN'T BEEN MERGED YET
    # merged_data = merge_data(meteo_data=df_meteo, ssm_data=df_ssm, days_num=DAYS_PER_SNIP)
    # csv_data = merged_data.to_csv(MERGED_DF_NUT_NAME, sep=";")

    # IF SSM AND METEO DATA WERE MERGED IN EXECUTIONS BEFORE
    merged_data = pd.read_csv(MERGED_DATASET, sep=";")
    redundant_column_name = "Unnamed: 0"
    del merged_data[redundant_column_name]

    # converting to numpy
    numpy_y = np.array(y_nut).flatten()
    numpy_merged_data = merged_data.to_numpy()

    # в данных 4262 записи, поэтому в дальнейшем будем брать size_source = 500, size_target = 100
    # source_size = 500
    # target_size = 100

    # тестовые значения, для отладки кода
    source_size = 100
    target_size = 20
    total_source_target = source_size + target_size

    parsed_data = parse_source_target(x=numpy_merged_data, y=numpy_y, source_size=source_size, target_size=target_size)
    res = ITGP(parsed_data[0], parsed_data[1], parsed_data[2], parsed_data[3], preload_models=True)


# if __name__ == 'main':
#     main()

main()
