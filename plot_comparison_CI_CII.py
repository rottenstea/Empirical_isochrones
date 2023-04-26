import my_utility

import seaborn as sns
import matplotlib.pyplot as plt

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list
from Empirical_iso_reader import build_empirical_df, general_kwargs, empirical_iso_path

output_path = my_utility.set_output_path()
save_plot = False

del general_kwargs["col_names"]
del general_kwargs["filename_exclude"]

merged_1 = build_empirical_df(filename_key="G_BPRP_nboot_1000_cat_1", col_names=["ref_age"], **general_kwargs)
merged_2 = build_empirical_df(filename_key="G_BPRP_nboot_1000_cat_2", col_names=["ref_age"], **general_kwargs)

sorted_clusters = merged_2["Cluster_id"].drop_duplicates().to_numpy()
merged_1 = merged_1[merged_1['Cluster_id'].isin(merged_2['Cluster_id'])]

sns.set_style("darkgrid")

CI, CII = cluster_df_list[0], cluster_df_list[1]
CII_names = cluster_name_list[1]

for n, cluster in enumerate(CII_names[:]):

    iso_CI = pd.read_csv(empirical_iso_path + "{}_G_BPRP_nboot_1000_cat_1.csv".format(cluster))
    iso_CII = pd.read_csv(empirical_iso_path + "{}_G_BPRP_nboot_1000_cat_2.csv".format(cluster))

    # 1. Create a class object for each cluster
    OC_1 = star_cluster(cluster, CI)
    OC_2 = star_cluster(cluster, CII)
    # 2. Create the CMD that should be used for the isochrone extraction
    OC_1.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])
    OC_2.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])

    comp_fig = plt.figure(figsize=(4, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))

    ax.scatter(OC_1.CMD[:, 0], OC_1.CMD[:, 1], marker="o", s=15, color="black", alpha=0.5, label="Catalog I")
    ax.scatter(OC_2.CMD[:, 0], OC_2.CMD[:, 1], marker="x", s=15, color="darkgray", alpha=0.5, label="Catalog II")
    ax.plot(iso_CI["m_x"], iso_CI["m_y"], color="red", label="CI")
    ax.plot(iso_CII["m_x"], iso_CII["m_y"], color="darkred", label="CII")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel(f"absolute {OC_1.CMD_specs['axes'][0]}")
    ax.set_xlabel(OC_1.CMD_specs["axes"][1])

    ax.set_title(cluster)
    plt.show()

    if save_plot:
        comp_fig.savefig(output_path + "{}_CI_vs_CII.pdf".format(cluster), dpi=500)
