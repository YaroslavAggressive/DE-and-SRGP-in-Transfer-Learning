import h5py
import pandas as pd
from pandas import DataFrame
import numpy as np

Y_KEY = 'response'
EXCEPTING_KEY = 'gr_names'
WEATHER_KEYS = ["tmin", "tmax", "srad", "rain", "dl"]

SSM_FILENAME = "datasets/chickpea-ssm-snp.h5"
METEO_FILENAME = "datasets/weather_ssm.h5"
MERGED_DATASET = "datasets/merged_weather_ssm.csv"


def get_data_response() -> DataFrame:
    ssm_data = h5py.File(SSM_FILENAME)
    y = ssm_data.get(Y_KEY)
    return DataFrame(y)


def get_ssm_data() -> DataFrame:

    ssm_data = h5py.File(SSM_FILENAME)
    y = ssm_data.get(Y_KEY)

    df_ssm = pd.DataFrame()
    df_ssm_keys = []
    tmp_column_names = []  # for separating different genes per column

    for key in ssm_data.keys():

        if key == Y_KEY or key == 'species':  # this column is the response, we will predict it with the model
            continue

        if key != EXCEPTING_KEY:  # so as not to add a column name that will not exist in reality
            df_ssm_keys.append(key)
        else:  # immediately process this case and go to the next iteration
            tmp_keys = ssm_data.get(key)[:]
            df_ssm_keys += list(tmp_keys)
            df_ssm = df_ssm.rename(
                columns={old_name: new_name for old_name, new_name in zip(tmp_column_names, tmp_keys)})
            continue

        ssm_column = ssm_data.get(key)
        if len(ssm_column.shape) < 2:
            df_ssm[key] = ssm_column[:]
        elif ssm_column.shape[1] > 1:
            for i in range(ssm_column.shape[1]):
                col_name = key + "_" + str(i)
                df_ssm[col_name] = ssm_column[:, i]
                tmp_column_names.append(col_name)
        else:
            df_ssm[key] = ssm_column[:, 0]

    return df_ssm


def get_meteo_data() -> DataFrame:
    meteo_data = h5py.File(METEO_FILENAME)
    df_meteo = pd.DataFrame()
    for key in meteo_data.keys():
        new_column = meteo_data.get(key)
        df_meteo[key] = new_column[:, 0]
    return df_meteo


def check_shift(df: DataFrame, shift: int, ind: int, days_num: int):
    for j in range(days_num):
        tmp_df = df[df["doy"] == ind + j + shift]
        if tmp_df.empty:
            return False
    return True


def merge_data(meteo_data: DataFrame, ssm_data: DataFrame, days_num: int, verbose: bool = False):
    merged_df = pd.DataFrame()
    dict_ssm_df = ssm_data.to_dict()
    ssm_keys = ssm_data.keys()
    for row in range(ssm_data.shape[0]):
        new_dict_row = dict()
        for key in ssm_keys:
            new_dict_row.update({key: dict_ssm_df[key][row]})

        tmp_geo_id = dict_ssm_df["geo_id"][row]  # for choosing correctly intervals of data reloading
        tmp_year = dict_ssm_df["year"][row]
        tmp_doy = dict_ssm_df["doy"][row]

        rows_subset = meteo_data[(meteo_data["geo_id"] == tmp_geo_id) &
                                 (meteo_data["year"] == tmp_year)]

        shift_subset = meteo_data[(meteo_data["geo_id"] == tmp_geo_id) &
                                  (meteo_data["year"] == tmp_year + 1)]

        shift = 0
        while tmp_doy + days_num > 365:
            shift = tmp_doy + days_num - 365
        for j in range(days_num):
            temp_dict = dict()
            for key in WEATHER_KEYS:
                if shift == 0:
                    temp_dict.update({key + str(j): [float(rows_subset[rows_subset["doy"] == tmp_doy + j][key])]})
                else:
                    temp_dict.update({key + str(j): [float(shift_subset[shift_subset["doy"] == j][key])]})
            new_dict_row.update(temp_dict)
        if verbose:
            print("Iteration #{}".format(row + 1))
        tmp_df = pd.DataFrame.from_dict(new_dict_row)
        merged_df = pd.concat([merged_df, tmp_df])

    return merged_df


def parse_train_test_valid(x_df: DataFrame, y_df: DataFrame, train_size: int, test_size: int, valid_size: int):
    if len(x_df) != len(y_df):
        raise Exception("Parameters and responses are inconsistent in size")
    df_size = x_df.shape[0]
    if df_size != train_size + test_size + valid_size:
        raise Exception("Parameters of train, test and validation datasets are inconsistent in size")
    indices = np.unique(df_size)

    source_indices = np.random.choice(indices, size=train_size, replace=False)
    other_indices = np.setdiff1d(indices, source_indices)
    target_indices = np.random.choice(other_indices, size=test_size, replace=False)
    validation_indices = np.setdiff1d(other_indices, target_indices)

    x_source, y_source = x_df[source_indices, :], y_df[source_indices]
    x_target, y_target = x_df[target_indices, :], y_df[target_indices]
    x_valid, y_valid = x_target[validation_indices, :], y_df[validation_indices]

    return [x_source, y_source, x_target, y_target, x_valid, y_valid]


def parse_source_target(x: np.array, y: np.array, source_size: int, target_size: int) -> list:
    if len(x) != len(y):
        raise Exception("Parameters and responses are inconsistent in size")
    dataset_size = len(x)
    indices = np.arange(dataset_size)
    if source_size + target_size > dataset_size:
        raise Exception("Too large sizes of training and target data")
    source_indices = np.random.choice(indices, size=source_size, replace=False)
    target_indices = np.random.choice(np.setdiff1d(indices, source_indices), size=target_size, replace=False)

    x_source, y_source = x[source_indices, :], y[source_indices]
    x_target, y_target = x[target_indices, :], y[target_indices]

    return [x_source, y_source, x_target, y_target]
