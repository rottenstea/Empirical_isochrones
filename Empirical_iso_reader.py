import os
import pandas as pd

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"

reference_ages = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Reference_ages.csv")


def build_empirical_df(csv_folder: str, age_file: pd.DataFrame, col_names: list, name_split: str = None,
                       filename_key: str = None, filename_exclude: str = None, filter_function=None):
    # get a list of all the csv files in the directory
    if filename_key:
        csv_list = [file for file in os.listdir(csv_folder) if filename_key in file]
    else:
        csv_list = [file for file in os.listdir(csv_folder)]

    if filename_exclude:
        csv_list = [x for x in csv_list if filename_exclude not in x]

    # read in all csv files and concatenate them into one dataframe
    df_list = [pd.read_csv(os.path.join(csv_folder, file)).assign(Cluster_id=file.split(name_split)[0]) for file in
               csv_list]
    # add extra filter criterion if not all dfs are wanted
    if filter_function:
        df_list = filter_function(df_list)

    concat_df = pd.concat(df_list, ignore_index=True)

    merged_df = pd.merge(concat_df, age_file, on='Cluster_id')
    merged_df.sort_values(by=col_names, inplace=True, ascending=True)

    return merged_df


# ----------------------------------------------------------------------------------------------------------------------

general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                      name_split="_G", filename_exclude="cat_2")

# BP - RP
# ========

merged_BPRP = build_empirical_df(filename_key="G_BPRP_nboot_1000", **general_kwargs)

# BP - G
# ========

merged_BPG = build_empirical_df(filename_key="G_BPG_nboot_1000", **general_kwargs)

# G - RP
# ========

merged_GRP = build_empirical_df(filename_key="G_GRP_nboot_1000", **general_kwargs)
