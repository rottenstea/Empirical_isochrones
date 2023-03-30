import pandas as pd
import numpy as np


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


# Catalog import
# ----------------------------------------------------------------------------------------------------------------------
# Cantat-Gaudin 2020 = CATALOG I
# ----------------------------------------------------------------------------------------------------------------------

CI_raw = "data/Cluster_data/all_ages/CatalogI_BCD_ages.csv"

CI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
           "G-RP",
           "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
           "RUWE", "Proba"]

CI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability"]

q_filter = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"], "value": [1.4, 0, 0.49]}

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter)

CI_df["ref_age"] = CI_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# M 2020 Xmatch Gaia EDR3 + errors == CATALOG II
# ----------------------------------------------------------------------------------------------------------------------

CII_raw = "data/Cluster_data/all_ages/CatalogII_BCD_ages.csv"

CII_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
            "RUWE"]

CII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

CII_clusters, CII_df = create_df(CII_raw, CII_cols, CII_names, q_filter)

CII_df["ref_age"] = CII_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# new catalogue from Sebastian == CATALOG III
# ----------------------------------------------------------------------------------------------------------------------

CIII_raw = "data/Cluster_data/all_ages/CatalogIII_DR3_Seb_ages.csv"

CIII_cols = ["cluster_name", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "logage_lts", "logage_tdist",
             "ruwe", "fidelity_v2", "stability", "G_err", "G_BPerr", "G_RPerr"]

CIII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
              "BP-G", "G-RP",
              "age_lts", "age_tdist", "ruwe", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"]

q_filter = {"parameter": ["ruwe", "plx", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"],
            "limit": ["upper", "lower", "lower", "lower", "upper", "upper", "upper"],
            "value": [1.4, 0, 0.5, 6, 0.007, 0.15, 0.03]}

CIII_clusters, CIII_df = create_df(CIII_raw, CIII_cols, CIII_names, q_filter)

CIII_df["ref_age"] = CIII_df["age_tdist"]

# ----------------------------------------------------------------------------------------------------------------------
# Coma Ber (Melotte 111) == ADD-ON I
# ----------------------------------------------------------------------------------------------------------------------

AOI_raw = "data/Cluster_data/all_ages/Coma_Ber_CD_ages.csv"

AOI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
            "RUWE"]

AOI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

AOI_clusters, AOI_df = create_df(AOI_raw, AOI_cols, AOI_names, q_filter)

AOI_df["ref_age"] = AOI_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# Hyades (Melotte 25) == ADD-ON II
# ----------------------------------------------------------------------------------------------------------------------

AOII_raw = "data/Cluster_data/all_ages/Hyades_CD_ages.csv"

AOII_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
             "G-RP",
             "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
             "RUWE"]

AOII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
              "BP-G", "G-RP",
              "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

AOII_clusters, AOII_df = create_df(AOII_raw, AOII_cols, AOII_names, q_filter)
AOII_df["ref_age"] = AOII_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# Meingast 1 == CASE STUDY I
# ----------------------------------------------------------------------------------------------------------------------

CSI_raw = "data/Cluster_data/all_ages/Meingast1_stab_24_CuESSIV_ages.csv"

CSI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "logage_Curtis", "logage_ESSIV",
            "RUWE", "Stab"]

CSI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_Cu", "age_ESSIV", "ruwe", "stability"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

CSI_clusters, CSI_df = create_df(CSI_raw, CSI_cols, CSI_names, q_filter)

CSI_df["ref_age"] = CSI_df["age_Cu"]

# ----------------------------------------------------------------------------------------------------------------------


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
