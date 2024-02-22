import os
import pandas as pd
from EmpiricalArchive.My_tools import my_utility

"""Script for reading in the empirical isochrones saved for each cluster and for creating a master-table containing
all results for the paper."""


def build_empirical_df(csv_folder: str, age_file: pd.DataFrame, col_names: list, name_split: str = None,
                       filename_key: str = None, filename_exclude: str = None, filter_function=None):
    """
    Read in the empirical isochrone and uncertainty bounds for each cluster from separate csv files and knit them into
    a single dataframe. Grabs the files specified by keys or filter functions.

    :param csv_folder: Location of the individual result csv-files.
    :param age_file: File containing the reference ages. They will be added to the empirical isochrone information.
    :param col_names: Columns by which the dataframe should be sorted.
    :param name_split: String upon which to split the filename to assign the cluster name to the respective isochrone.
    :param filename_key: String that needs to be included in the filename, e.g. catalog number or CMD combi.
    :param filename_exclude: Files with this string will not be included in the dataframe. Opposite of filename_key.
    :param filter_function: User-given function that one can apply to the list of dataframes and that gives out an
                            updated list.
    :return: Dataframe
    """

    # get a list of all the csv files in the directory, using include or exclude statements if they are defined
    if filename_key:
        csv_files = [file for file in os.listdir(csv_folder) if filename_key in file]
    else:
        csv_files = [file for file in os.listdir(csv_folder)]

    if filename_exclude:
        csv_files = [x for x in csv_files if filename_exclude not in x]

    # read in all csv files and concatenate them into one dataframe
    single_dfs = [pd.read_csv(os.path.join(csv_folder, file)).assign(Cluster_id=file.split(name_split)[0]) for file in
                  csv_files]

    # add extra filter criterion with an arbitrary function if not all dfs are wanted
    if filter_function:
        single_dfs = filter_function(single_dfs)

    # concat the list of single dataframes into a large one
    concatenated_df = pd.concat(single_dfs, ignore_index=True)

    # merge the df with the provided age file using the cluster name as identifier
    results_and_age_df = pd.merge(concatenated_df, age_file, on='Cluster_id')

    # sort the dataframe by the specified columns
    results_and_age_df.sort_values(by=col_names, inplace=True, ascending=True)

    return results_and_age_df


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    # set paths
    empirical_iso_path = "//data/Isochrones/Empirical/"
    reference_ages = pd.read_csv("//data/Isochrones/Empirical/")
    output_path = my_utility.set_output_path()

    save_table = True
    # ----------------------------------------------------------------------------------------------------------------------
    # Result dfs

    # Set general kwargs for most of the files
    general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                          name_split="_G")

    # create general result dfs for the passband combinations
    merged_BPRP = build_empirical_df(filename_key="G_BPRP_nboot_1000", **general_kwargs)  # BP - RP
    merged_BPG = build_empirical_df(filename_key="G_BPG_nboot_1000", **general_kwargs)  # BP - G
    merged_GRP = build_empirical_df(filename_key="G_GRP_nboot_1000", **general_kwargs)  # G - RP
    # ----------------------------------------------------------------------------------------------------------------------
    # Mastertable Gaia

    # exclude the MG1 ESS 2 case study from the age_reference file
    ref_ages_mastertable = reference_ages[reference_ages["Cluster_id"] != "Meingast_1_ESSII"]

    # only use cluster identifier and reference age column
    ref_ages_mastertable = ref_ages_mastertable[["Cluster_id", "ref_age"]]

    # list for collecting dfs
    dfs = []

    # iterate through the clusters in the age reference df
    for cluster in sorted(ref_ages_mastertable["Cluster_id"]):

        # collect the result files for that cluster name
        csv_unsorted = [file for file in os.listdir(empirical_iso_path) if "{}_G".format(cluster) in file]

        # sort the files (has to be done in reverse)
        sorted_csvs = sorted(csv_unsorted, key=lambda x: ('GRP' in x, 'BPG' in x, 'BPRP' in x))

        # make a list of df (three dfs for the three combinations)
        df_list = [pd.read_csv(os.path.join(empirical_iso_path, file)).assign(Cluster_id=cluster) for file in
                   sorted_csvs]

        # the dfs all have the same column names, for right merging into the mastertable a prefix needs to be added to them
        bands = ["BPRP_", "BPG_", "GRP_"]
        renamed_list = []

        for idx, df in enumerate(df_list):

            # rename all columns but the identifier
            df = df.iloc[:, [1, 2, 3, 4, 5, 6]]
            df_renamed = df.add_prefix(bands[idx])
            renamed_list.append(df_renamed)

        # concat the three dataframes
        all_passbands_df = pd.concat(renamed_list, axis=1)

        # insert the identifier column again at index zero
        all_passbands_df.insert(0, "Cluster_id", cluster)

        # add the concatenated dataframe for the cluster instance to the df list
        dfs.append(all_passbands_df)

    # create the mastertable from the dataframes for all Gaia clusters
    mastertable_dfs = pd.concat(dfs, ignore_index=True)
    mastertable = pd.merge(mastertable_dfs, ref_ages_mastertable, on='Cluster_id')

    # rename the columns
    official_names = ['Cluster', 'BPRP_lb_x', 'BPRP_lb_y', 'BPRP_isochrone_x', 'BPRP_isochrone_y', 'BPRP_ub_x', 'BPRP_ub_y',
                      'BPG_lb_x', 'BPG_lb_y', 'BPG_isochrone_x', 'BPG_isochrone_y', 'BPG_ub_x', 'BPG_ub_y',
                      'GRP_lb_x', 'GRP_lb_y', 'GRP_isochrone_x', 'GRP_isochrone_y', 'GRP_ub_x', 'GRP_ub_y', 'ref_age']
    mastertable.columns = official_names

    # save the table
    if save_table:
        mastertable.to_csv("/Users/alena/PycharmProjects/Empirical_Isochrones/data/Isochrones/Mastertable_Archive.csv",
                           mode="w", header=True)
    # ----------------------------------------------------------------------------------------------------------------------