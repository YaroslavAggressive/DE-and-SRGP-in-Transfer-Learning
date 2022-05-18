from dataset_parsing import initial_parse_data_and_save, MERGED_DATASET
from dataset_parsing import get_data_response, parse_per_snp, parse_per_season, parse_per_key, DATASET_SEASONS
import pandas as pd

initial_parse_data_and_save()

merged_dataset = pd.read_csv(MERGED_DATASET, sep=';')
y_merged = get_data_response()
redundant_column_name = "Unnamed: 0"
del merged_dataset[redundant_column_name]

# creating datasets per snp (missing elements which include chosen snips)
for i in range(6):
    snp_ind = i + 1
    dataset_data = parse_per_snp(merged_dataset, y_merged, snp_ind)
    dataset_data[0].to_csv("snp_datasets/data_no_snp_" + str(snp_ind))
    dataset_data[1].to_csv("snp_datasets/data_no_snp_" + str(snp_ind) + "_response")

# creating datasets per season
for season in DATASET_SEASONS.keys():
    dataset_data = parse_per_season(merged_dataset, y_merged, season)
    dataset_data[0].to_csv("seasons_datasets/data_season_" + season)
    dataset_data[1].to_csv("seasons_datasets/data_season_" + season + "_response")

# creating datasets per geo_id

for geo_id in [0., 2., 3.]:
    dataset_data = parse_per_key(merged_dataset, y_merged, geo_id, "geo_id")
    dataset_data[0].to_csv("geo_datasets/data_geo_" + str(geo_id))
    dataset_data[1].to_csv("geo_datasets/data_geo_" + str(geo_id) + "_response")
