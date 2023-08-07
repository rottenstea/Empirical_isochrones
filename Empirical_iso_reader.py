import os
import pandas as pd
import my_utility

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"

reference_ages = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Reference_ages.csv")

output_path = my_utility.set_output_path()


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
                      name_split="_G")

# BP - RP
# ========

merged_BPRP = build_empirical_df(filename_key="G_BPRP_nboot_1000", **general_kwargs)

# BP - G
# ========

merged_BPG = build_empirical_df(filename_key="G_BPG_nboot_1000", **general_kwargs)

# G - RP
# ========

merged_GRP = build_empirical_df(filename_key="G_GRP_nboot_1000", **general_kwargs)

all_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                  name_split="_G")

# ----------------------------------------------------------------------------------------------------------------------
# MASTERTABLE
# ===========

ref_ages_mastertable = reference_ages[reference_ages["Cluster_id"] != "Meingast_1_ESSII"]
ref_ages_mastertable = ref_ages_mastertable[["Cluster_id", "ref_age"]]
dfs = []

for cluster in sorted(ref_ages_mastertable["Cluster_id"]):
    csv_list = [file for file in os.listdir(empirical_iso_path) if "{}_G".format(cluster) in file]
    df_list = [pd.read_csv(os.path.join(empirical_iso_path, file)).assign(Cluster_id=cluster) for file in
               csv_list]
    bands = ["BPRP_", "BPG_", "GRP_"]
    renamed_list = []

    for idx, df in enumerate(df_list):
        df = df.iloc[:, [1, 2, 3, 4, 5, 6]]
        df_renamed = df.add_prefix(bands[idx])
        renamed_list.append(df_renamed)

    concat_df = pd.concat(renamed_list, axis=1)
    concat_df.insert(0, "Cluster_id", cluster)
    dfs.append(concat_df)
    # concat_df.to_csv(output_path + "test_{}.csv".format(cluster), mode="w", header=True)

concat_large = pd.concat(dfs, ignore_index=True)
merged_df = pd.merge(concat_large, ref_ages_mastertable, on='Cluster_id')
official_names = ['Cluster', 'BPRP_lb_x', 'BPRP_lb_y', 'BPRP_isochrone_x', 'BPRP_isochrone_y', 'BPRP_ub_x', 'BPRP_ub_y',
                  'BPG_lb_x', 'BPG_lb_y', 'BPG_isochrone_x', 'BPG_isochrone_y', 'BPG_ub_x', 'BPG_ub_y',
                  'GRP_lb_x', 'GRP_lb_y', 'GRP_isochrone_x', 'GRP_isochrone_y', 'GRP_ub_x', 'GRP_ub_y', 'ref_age']
merged_df.columns = official_names
# merged_df.sort_values(by=col_names, inplace=True, ascending=True)

merged_df.to_csv(output_path + "Mastertable_Archive.csv", mode="w", header=True)
