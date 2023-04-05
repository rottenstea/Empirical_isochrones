import pandas as pd
import numpy as np

"""
This file holds the function that converts the raw csv input files into pd.DataFrames and only retains the necessary
columns. It also contains the code that does just this transformation for the 8 different catalogs for the paper so
that they may just be imported into the various python scripts by their variable name or the variable name of the list.
"""

data_path = "/Users/alena/PycharmProjects/PaperI/"

def create_df(filepath, columns, names, quality_filter: dict = None):
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


# Archive import
# ======================================================================================================================

standard_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
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

CI_cols = standard_cols + ["logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "Proba", "X", "Y", "Z",
                           "N"]

CI_names = standard_names + ["age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability", "x", "y", "z",
                             "Nstars"]

q_filter = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"], "value": [1.4, 0, 0.49]}

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter)
CI_df["catalog"] = 1
CI_df["ref_age"] = CI_df["age_C"]

cluster_df_list.append(CI_df)
cluster_name_list.append(CI_clusters)

# ----------------------------------------------------------------------------------------------------------------------
# M 2020 Xmatch Gaia EDR3 + errors == CATALOG II
# ----------------------------------------------------------------------------------------------------------------------

CII_raw = data_path + "data/Cluster_data/all_ages/CatalogII_BCD_ages.csv"

CII_cols = standard_cols + ["logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE", "X_CG", "Y_CG", "Z"]

CII_names = standard_names + ["age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "x", "y", "z"]

q_filter = standard_filter

CII_clusters, CII_df = create_df(CII_raw, CII_cols, CII_names, q_filter)

CII_df["catalog"] = 2
CII_df["ref_age"] = CII_df["age_C"]

cluster_df_list.append(CII_df)
cluster_name_list.append(CII_clusters)

# ----------------------------------------------------------------------------------------------------------------------
# new catalogue from Sebastian == CATALOG III
# ----------------------------------------------------------------------------------------------------------------------

CIII_raw = data_path + "data/Cluster_data/all_ages/CatalogIII_DR3_Seb_ages.csv"

CIII_cols = standard_cols + ["logage_lts", "logage_tdist", "ruwe", "fidelity_v2", "stability", "G_err", "G_BPerr",
                             "G_RPerr"]

CIII_names = standard_names + ["age_lts", "age_tdist", "ruwe", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"]

q_filter = {"parameter": ["ruwe", "plx", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"],
            "limit": ["upper", "lower", "lower", "lower", "upper", "upper", "upper"],
            "value": [1.4, 0, 0.5, 6, 0.007, 0.05, 0.03]}

CIII_clusters, CIII_df = create_df(CIII_raw, CIII_cols, CIII_names, q_filter)

CIII_df["catalog"] = 3
CIII_df["ref_age"] = CIII_df["age_tdist"]

# ----------------------------------------------------------------------------------------------------------------------
# Coma Ber (Melotte 111) == ADD-ON I
# ----------------------------------------------------------------------------------------------------------------------

AOI_raw = data_path + "data/Cluster_data/all_ages/Coma_Ber_CD_ages.csv"

AOI_cols = standard_cols + ["AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE"]

AOI_names = standard_names + ["age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = standard_filter

AOI_clusters, AOI_df = create_df(AOI_raw, AOI_cols, AOI_names, q_filter)

AOI_df["catalog"] = 4
AOI_df["ref_age"] = AOI_df["age_C"]

cluster_df_list.append(AOI_df)
cluster_name_list.append(AOI_clusters)

# ----------------------------------------------------------------------------------------------------------------------
# Hyades (Melotte 25) == ADD-ON II
# ----------------------------------------------------------------------------------------------------------------------

AOII_raw = data_path + "data/Cluster_data/all_ages/Hyades_CD_ages.csv"

AOII_cols = standard_cols + ["AgeNN_CG", "AVNN_CG", "logage_D", "Av_D", "RUWE"]

AOII_names = standard_names + ["age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = standard_filter

AOII_clusters, AOII_df = create_df(AOII_raw, AOII_cols, AOII_names, q_filter)

AOII_df["catalog"] = 5
AOII_df["ref_age"] = AOII_df["age_C"]

cluster_df_list.append(AOII_df)
cluster_name_list.append(AOII_clusters)

# ----------------------------------------------------------------------------------------------------------------------
# Meingast 1 == ADD-ON III / CASE STUDY I
# ----------------------------------------------------------------------------------------------------------------------

AOIII_raw = data_path + "data/Cluster_data/all_ages/Meingast1_stab_24_CuESSIV_ages.csv"

AOIII_cols = standard_cols + ["logage_Curtis", "logage_ESSIV", "RUWE", "Stab"]

AOIII_names = standard_names + ["age_Cu", "age_ESSIV", "ruwe", "stability"]

q_filter = standard_filter

AOIII_clusters, AOIII_df = create_df(AOIII_raw, AOIII_cols, AOIII_names, q_filter)

AOIII_df["catalog"] = 5
AOIII_df["ref_age"] = AOIII_df["age_Cu"]

cluster_df_list.append(AOIII_df)
cluster_name_list.append(AOIII_clusters)

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

q_filter = {"parameter": ["probability", "e_imag"], "limit": ["lower", "upper"], "value": [0.84, 0.3]}

CSII_cluster, CSII_df = create_df(CSII_raw, CSII_cols, CSII_names, q_filter)

CSII_df["ref_age"] = CSII_df["age_C"]

case_study_dfs.append(CSII_df)

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

q_filter = {"parameter": ["probability"], "limit": ["lower"], "value": [0.5]}

CSIII_cluster, CSIII_df = create_df(CSIII_raw, CSIII_cols, CSIII_names, q_filter)

CSIII_df["ref_age"] = CSIII_df["age_C"]

# here NaN are actually 99
CSIII_df.replace(float(99), np.nan, inplace=True)

case_study_dfs.append(CSIII_df)

# ======================================================================================================================


# testing
# if __name__ == "__main__":
#
#     CSIII_raw = "data/Cluster_data/all_ages/IC_4665_BCD_ages.csv"
#
#     CSIII_cols = ["Cluster", "median_plx", "g", "r", "i", "z", "y", "J", "H", "K",
#                   "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
#                   "prob_max"]
#
#     CSIII_names = ["Cluster_id", "plx", "gmag", "rmag", "imag", "zmag", "ymag", "Jmag", "Hmag", "Kmag",
#                    "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "probability"]
#
#     q_filter = {"parameter": ["probability"], "limit": ["lower"], "value": [0.5]}
#
#     CSIII_cluster, CSIII_df = create_df(CSIII_raw, CSIII_cols, CSIII_names, q_filter)
#
#     print(min(CSIII_df.probability))
