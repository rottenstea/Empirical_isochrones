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


# testing
if __name__ == "__main__":

    CSIII_raw = "data/Cluster_data/all_ages/IC_4665_BCD_ages.csv"

    CSIII_cols = ["Cluster", "median_plx", "g", "r", "i", "z", "y", "J", "H", "K",
                  "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
                  "prob_max"]

    CSIII_names = ["Cluster_id", "plx", "gmag", "rmag", "imag", "zmag", "ymag", "Jmag", "Hmag", "Kmag",
                   "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "probability"]

    q_filter = {"parameter": ["probability"], "limit": ["lower"], "value": [0.5]}

    CSIII_cluster, CSIII_df = create_df(CSIII_raw, CSIII_cols, CSIII_names, q_filter)

    print(min(CSIII_df.probability))
