import h5py
import pandas as pd
from pandas import DataFrame
import numpy as np

Y_KEY = 'response'
EXCEPTING_KEY = 'gr_names'
WEATHER_KEYS = ["tmin", "tmax", "srad", "rain", "dl"]

SSM_FILENAME = "datasets/chickpea-ssm-snp.h5"
METEO_FILENAME = "datasets/weather_ssm.h5"
MERGED_DATASET = "datasets/merged_weather_ssm.csv"  # 25.05.22 - убран ненужный параметр 'month'

SEASON_KEY = "month"  # for meaningful separation of data for model development and testing
GEO_ID_KEY = "geo_id"
DOY_KEY = "doy"
DATASET_SEASONS = {"winter": [1, 2, 12], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}
SNP_KEYS = {1: [2, 3, 4], 2: [5, 6, 7], 3: [8, 9, 10], 4: [11, 12, 13], 5: [14, 15, 16], 6: [17, 18, 19]}
SUFFIXES = ["AA", "AR", "RR"]

DATASET_SNP_PREFIX = "dataset_by_snp"
DATASET_SEASON_PREFIX = "dataset_by_season"
DATASET_GEO_PREFIX = "dataset_by_geo"

DAYS_AGEEV_DATASET = [290.0, 294.0, 307.0, 325.0, 339.0]


def get_data_response() -> DataFrame:
    ssm_data = h5py.File(SSM_FILENAME)
    y = ssm_data.get(Y_KEY)
    return DataFrame(y, columns=["response"])


def get_ssm_data() -> DataFrame:

    ssm_data = h5py.File(SSM_FILENAME)

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
            if key != "month":
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


def parse_x_y(x: np.array, y: np.array, size: int) -> list:
    if len(x) != len(y):
        raise Exception("Parameters and responses are inconsistent in size")
    dataset_size = len(x)
    indices = np.arange(dataset_size)
    chosen_indices = np.random.choice(indices, size=size, replace=False)

    return [x[chosen_indices, :], y[chosen_indices]]


def parse_per_key(x_df: DataFrame, y_df: DataFrame, key_val: float, key: str) -> list:
    x = x_df[x_df[key] == key_val]
    y = y_df.loc[x.index]
    return [x, y]


def parse_per_season(x_df: DataFrame, y_df: DataFrame, season: str) -> list:
    season_months = DATASET_SEASONS[season]

    x = x_df[(x_df[SEASON_KEY] == season_months[0]) |
             (x_df[SEASON_KEY] == season_months[1]) |
             (x_df[SEASON_KEY] == season_months[2])]

    y = y_df.loc[x.index]
    return [x, y]


def create_snp_columns(ind: int) -> list:
    res = []
    for suffix in SUFFIXES:
        column_name = "b'snp" + str(ind) + suffix + "'"
        res.append(column_name)
    return res


def parse_per_snp(x_df: DataFrame, y_df: DataFrame, snp: int) -> list:
    source_columns = create_snp_columns(snp)

    x = x_df[(x_df[source_columns[0]] == 0.) &
             (x_df[source_columns[1]] == 0.) &
             (x_df[source_columns[2]] == 0.)]
    source_indices = x.index
    y = y_df.loc[source_indices]

    return [x, y]


def parse_autumn_spring(x_df: DataFrame, y_df: DataFrame):  # here we parse source data, of course
    x_src = x_df[(x_df[DOY_KEY] == DAYS_AGEEV_DATASET[0]) |
                 (x_df[DOY_KEY] == DAYS_AGEEV_DATASET[1]) |
                 (x_df[DOY_KEY] == DAYS_AGEEV_DATASET[2]) |
                 (x_df[DOY_KEY] == DAYS_AGEEV_DATASET[3]) |
                 (x_df[DOY_KEY] == DAYS_AGEEV_DATASET[4])]
    indices_src = x_src.index
    y_src = y_df.loc[indices_src]

    all_indices = x_df.index
    indices_trg = np.setdiff1d(all_indices, indices_src)
    x_trg, y_trg = x_df.loc[indices_trg], y_df.loc[indices_trg]
    return [x_src, y_src, x_trg, y_trg]


def parse_valid(x_df: DataFrame, y_df: DataFrame, size: int):
    x_df_shuffle = x_df.sample(n=size)
    indices = x_df_shuffle.index
    y_df_shuffle = y_df.loc[indices]
    return [x_df_shuffle, y_df_shuffle]


def data_shuffle(x_df: DataFrame, y_df: DataFrame):
    x_shuffle = x_df.sample(frac=1)
    y_shuffle = y_df.loc[x_shuffle.index]
    return [x_shuffle, y_shuffle]


gg = get_data_response()
gg.to_csv("datasets/response_chickpea.csv", index=True, sep=";")
