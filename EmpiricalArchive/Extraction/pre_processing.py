import numpy as np
import pandas as pd

"""
This file holds the function that converts the raw csv input files into pd.DataFrames and only retains the necessary
columns. It also contains the code that does just this transformation for the 8 different catalogs for the paper so
that they may just be imported into the various python scripts by their variable name or the variable name of the list.
"""


def create_df(filepath, columns, names, quality_filter: dict = None):
    """
    Read in and reformat the input data.

    :param filepath: Path to csv-file
    :param columns: columns to include in the import
    :param names: uniform names for the columns on which the code works
    :param quality_filter: dictionary of additional quality filters. Requires the parameter name (= column name), limit type (upper/lower) and the numerical filter value
    :return: namelist of all clusters in the dataframe and the formatted dataframe
    """
    raw_df = pd.read_csv(filepath)

    # infer list of cluster names directly from raw data
    cluster_list = raw_df[columns[0]].unique()

    # only incorporate columns holding CMD relevant data
    catalog = raw_df[columns]

    # use standardised names to make code run smoothly
    catalog.columns = names

    # impose more strict cuts
    if quality_filter:

        for q, fil in enumerate(quality_filter["parameter"]):
            c_mask = np.where(quality_filter["limit"][q] == "lower", catalog[fil] > quality_filter["value"][q],
                              catalog[fil] < quality_filter["value"][q])  #
            catalog = catalog[c_mask]

    return cluster_list, catalog


def create_reference_csv(df_list: list, output_path: str, ref_key: str, master_ref: list = None,
                         id_col: str = "Cluster_id"):
    """
    As often different literature ages are available for the clusters, this function is designed to convert these
    into a csv-file and (optionally) choose a reference age based on the user input.

    :param df_list: List of input dataframes (usually corresponding to the different catalogs
    :param output_path: Location where the new csv-file should be saved
    :param ref_key: age / av
    :param master_ref: List of references ordered by their importance. The first will be preferred when chosing the ref age, then the second if the first is a NaN and so on, maximum = 3
    :param id_col: column containting cluster names or other identifiers
    :return:
    """
    concat_df = pd.concat(df_list, axis=0)
    concat_df.drop_duplicates(id_col, inplace=True)
    reference_cols = [id_col] + [col for col in concat_df.columns if ref_key in col]

    # for defining the reference column ad hoc
    if master_ref:
        concat_df["ref_{}".format(ref_key)] = np.where(~concat_df[master_ref[0]].isna(), concat_df[master_ref[0]],
                                                       np.where(~concat_df[master_ref[1]].isna(),
                                                                concat_df[master_ref[1]], concat_df[master_ref[2]]))
        reference_cols = ["ref_{}".format(ref_key)] + reference_cols

    # for just grabbing every column that has the keyword in it and creating a new file with them
    concat_df[reference_cols].to_csv(output_path, mode="w",
                                     header=True)


# test
# create_reference_csv(df_list=cluster_df_list,
# output_path="/Users/alena/PycharmProjects/Empirical_Isochrones/data/Reference_ages_new.csv", ref_key="age")

# ----------------------------------------------------------------------------------------------------------------------
# set path
data_path = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/"

# Archive import
standard_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
                 "BP-G",
                 "G-RP"]

standard_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
                  "BP-G", "G-RP"]

standard_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

cluster_df_list = []
cluster_name_list = []
# ----------------------------------------------------------------------------------------------------------------------
# Cantat-Gaudin 2020 = CATALOG I
# ----------------------------------------------------------------------------------------------------------------------
CI_raw = data_path + "data/Cluster_data/all_ages/CatalogI_BCD_ages.csv"

CI_cols = standard_cols + ["logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "Proba", "X", "Y",
                           "Z",
                           "N", "Plx_DR2", "Gmag_DR2", "BP-RP_DR2", "E(BP/RP)"]

CI_names = standard_names + ["age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability", "x", "y",
                             "z",
                             "Nstars", "plx_DR2", "Gmag_DR2", "BP-RP_DR2", "excess"]

q_filter_I = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"],
              "value": [1.4, 0, 0.49]}

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter_I)
CI_df["catalog"] = 1
CI_df["ref_age"] = np.where(~CI_df['age_C'].isna(), CI_df['age_C'],
                            np.where(~CI_df['age_D'].isna(), CI_df['age_D'], CI_df['age_B']))

CI_df = CI_df[(CI_df["Cluster_id"] != "IC_348") &
              (CI_df["Cluster_id"] != "L_1641S") &
              (CI_df["Cluster_id"] != "RSG_7")]

CI_clusters_new = CI_df["Cluster_id"].unique()

# print(CI_df.columns)
cluster_df_list.append(CI_df)
cluster_name_list.append(CI_clusters_new)
# ----------------------------------------------------------------------------------------------------------------------
# M 2020 Xmatch Gaia EDR3 + errors == CATALOG II
# ----------------------------------------------------------------------------------------------------------------------
CII_raw = data_path + "data/Cluster_data/all_ages/CatalogII_BCD_ages.csv"

CII_cols = standard_cols + ["logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "X_CG", "Y_CG",
                            "Z"]

CII_names = standard_names + ["age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "x", "y", "z"]

CII_clusters, CII_df = create_df(CII_raw, CII_cols, CII_names, standard_filter)

value_counts = CII_df['Cluster_id'].value_counts()
CII_df['Nstars'] = CII_df['Cluster_id'].map(value_counts)

CII_df["catalog"] = 2
CII_df["ref_age"] = CII_df["age_C"]

cluster_df_list.append(CII_df)
cluster_name_list.append(CII_clusters)
# ----------------------------------------------------------------------------------------------------------------------
# new catalogue from Sebastian == CATALOG III
# ----------------------------------------------------------------------------------------------------------------------
CIII_raw = data_path + "data/Cluster_data/all_ages/CatalogIII_DR3_Sco-Cen-ages-names_ages.csv"

CIII_cols = standard_cols + ["logage_lts", "logage_tdist", "ruwe", "fidelity_v2", "stability", "G_err", "G_BPerr",
                             "G_RPerr", "X", "Y", "Z", "plx_DR2", "Gmag_DR2", "BP-RP_DR2"]

CIII_names = standard_names + ["age_lts", "age_tdist", "ruwe", "fidelity", "stability", "G_err", "G_BPerr",
                               "G_RPerr",
                               "x", "y", "z", "plx_DR2", "Gmag_DR2", "BP-RP_DR2"]

q_filter_III = {"parameter": ["ruwe", "plx", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"],
                "limit": ["upper", "lower", "lower", "lower", "upper", "upper", "upper"],
                "value": [1.4, 0, 0.5, 6, 0.007, 0.05, 0.03]}

CIII_clusters, CIII_df = create_df(CIII_raw, CIII_cols, CIII_names, q_filter_III)

value_counts = CIII_df['Cluster_id'].value_counts()
CIII_df['Nstars'] = CIII_df['Cluster_id'].map(value_counts)
CIII_df = CIII_df[CIII_df["Nstars"] >= 100]

CIII_df["catalog"] = 3
CIII_df["ref_age"] = round(CIII_df["age_tdist"], 2)

CIII_df = CIII_df[(CIII_df["Cluster_id"] != "rho Oph/L1688") & (CIII_df["Cluster_id"] != "Lupus 1-4")]

CIII_clusters_new = CIII_df["Cluster_id"].unique()

cluster_df_list.append(CIII_df)
cluster_name_list.append(CIII_clusters_new)
# ----------------------------------------------------------------------------------------------------------------------
# Coma Ber (Melotte 111) == ADD-ON I
# ----------------------------------------------------------------------------------------------------------------------
AOI_raw = data_path + "data/Cluster_data/all_ages/Coma_Ber_CD_ages.csv"

AOI_cols = standard_cols + ["AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "X", "Y", "Z"]

AOI_names = standard_names + ["age_C", "av_C", "age_D", "av_D", "ruwe", "x", "y", "z"]

AOI_clusters, AOI_df = create_df(AOI_raw, AOI_cols, AOI_names, standard_filter)

AOI_df["catalog"] = 4
AOI_df["ref_age"] = AOI_df["age_C"]
AOI_df["Nstars"] = len(AOI_df["age_C"])

cluster_df_list.append(AOI_df)
cluster_name_list.append(AOI_clusters)
# ----------------------------------------------------------------------------------------------------------------------
# Hyades (Melotte 25) == ADD-ON II
# ----------------------------------------------------------------------------------------------------------------------
AOII_raw = data_path + "data/Cluster_data/all_ages/Hyades_CD_ages.csv"

AOII_cols = standard_cols + ["AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "x", "y", "z"]

AOII_names = standard_names + ["age_C", "av_C", "age_D", "av_D", "ruwe", "x", "y", "z"]

AOII_clusters, AOII_df = create_df(AOII_raw, AOII_cols, AOII_names, standard_filter)

AOII_df["catalog"] = 5
AOII_df["ref_age"] = AOII_df["age_C"]
AOII_df["Nstars"] = len(AOII_df["age_C"])

cluster_df_list.append(AOII_df)
cluster_name_list.append(AOII_clusters)
# ----------------------------------------------------------------------------------------------------------------------
# Meingast 1 == ADD-ON III / CASE STUDY I
# ----------------------------------------------------------------------------------------------------------------------
AOIII_raw = data_path + "data/Cluster_data/all_ages/Meingast1_stab_24_CuESSIV_ages.csv"

AOIII_cols = standard_cols + ["logage_Curtis", "logage_ESSIV", "RUWE", "Stab", "X", "Y", "Z"]

AOIII_names = standard_names + ["age_Cu", "age_ESSIV", "ruwe", "stability", "x", "y", "z"]

AOIII_filter = {"parameter": ["ruwe", "plx", "stability", "e_RPmag"], "limit": ["upper", "lower", "lower", "upper"],
                "value": [1.4, 0, 49.5, 0.03]}

AOIII_clusters, AOIII_df = create_df(AOIII_raw, AOIII_cols, AOIII_names, AOIII_filter)

AOIII_df["catalog"] = 5
AOIII_df["ref_age"] = AOIII_df["age_Cu"]
AOIII_df["Nstars"] = len(AOIII_df["age_Cu"])

cluster_df_list.append(AOIII_df)
cluster_name_list.append(AOIII_clusters)

AOIV_raw = data_path + "data/Cluster_data/all_ages/Meingast1_ESSII_DR3.csv"

AOIV_cols = standard_cols + ["logage_Curtis", "logage_ESSIV", "RUWE", "X_DR2", "Y_DR2", "Z_DR2"]

AOIV_names = standard_names + ["age_Cu", "age_ESSIV", "ruwe", "x", "y", "z"]

AOIV_filter = {"parameter": ["ruwe", "plx", "e_RPmag"], "limit": ["upper", "lower", "upper"],
               "value": [1.4, 0, 0.03]}

AOIV_clusters, AOIV_df = create_df(AOIV_raw, AOIV_cols, AOIV_names, AOIV_filter)
AOIV_df["catalog"] = 50
AOIV_df["ref_age"] = AOIV_df["age_Cu"]
AOIV_df["Nstars"] = len(AOIV_df["age_Cu"])

cluster_df_list.append(AOIV_df)
cluster_name_list.append(AOIV_clusters)
# ======================================================================================================================
# Case studies

case_study_dfs = []
case_study_names = ["Melotte_22", "IC_4665"]
# ----------------------------------------------------------------------------------------------------------------------
# Nuria pleiades cluster (DANCe) == CASE STUDY II
# ----------------------------------------------------------------------------------------------------------------------
CSII_raw = data_path + "data/Cluster_data/all_ages/Pleiades_BCD_ages.csv"

CSII_cols = ["Cluster", "mean_plx", "umag", "e_umag", "gmag", "e_gmag", "rmag", "e_rmag", "imag", "e_imag", "zmag",
             "e_zmag", "Ymag", "e_Ymag", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
             "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
             "pc"]

CSII_names = ["Cluster_id", "plx", "umag", "e_umag", "gmag", "e_gmag", "rmag", "e_rmag", "imag", "e_imag", "zmag",
              "e_zmag", "ymag", "e_ymag", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
              "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "probability"]

q_filter_CSII = {"parameter": ["probability", "e_imag", "zmag"], "limit": ["lower", "upper", "lower"],
                 "value": [0.839, 0.3, 0]}

CSII_cluster, CSII_df = create_df(CSII_raw, CSII_cols, CSII_names, q_filter_CSII)

CSII_df["ref_age"] = CSII_df["age_C"]

case_study_dfs.append(CSII_df)

# print("Pleiades:", CSII_df.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Nuria IC_4665 cluster (DANCe) == CASE STUDY III
# ----------------------------------------------------------------------------------------------------------------------
CSIII_raw = data_path + "data/Cluster_data/all_ages/IC_4665_BCD_ages.csv"

CSIII_cols = ["Cluster", "median_plx", "g", "e_g", "r", "e_r", "i", "e_i", "z", "e_z", "y", "e_y", "J", "e_J", "H",
              "e_H", "K", "e_K",
              "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
              "prob_max"]

CSIII_names = ["Cluster_id", "plx", "gmag", "e_gmag", "rmag", "e_rmag", "imag", "e_imag", "zmag", "e_zmag", "ymag",
               "e_ymag",
               "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
               "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "probability"]

q_filter_CSIII = {"parameter": ["probability", "rmag"], "limit": ["lower", "lower"], "value": [0.5, 9]}

CSIII_cluster, CSIII_df = create_df(CSIII_raw, CSIII_cols, CSIII_names, q_filter_CSIII)

CSIII_df["ref_age"] = CSIII_df["age_C"]

# here NaN are actually 99
CSIII_df.replace(float(99), np.nan, inplace=True)

case_study_dfs.append(CSIII_df)

# print("IC 4665:", CSIII_df.shape)
# ----------------------------------------------------------------------------------------------------------------------
