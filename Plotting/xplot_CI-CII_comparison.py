import seaborn as sns
import matplotlib.pyplot as plt

from My_tools import my_utility
from Extraction.Classfile import *
from Extraction.pre_processing import cluster_df_list, cluster_name_list
from Extraction.Empirical_iso_reader import build_empirical_df

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Load the results for Catalog I and II
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"
reference_ages = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Reference_ages.csv")
general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages)

# build two different dataframes with the results
merged_1 = build_empirical_df(filename_key="G_BPRP_nboot_1000", col_names=["ref_age"], name_split="_G",
                              **general_kwargs)
merged_2 = build_empirical_df(filename_key="C2_G_BPRP_nboot_1000", col_names=["ref_age"], name_split="_C2_G",
                              **general_kwargs)

# List of sorted clusternames
sorted_clusters = merged_2["Cluster_id"].drop_duplicates().to_numpy()
merged_1 = merged_1[merged_1['Cluster_id'].isin(merged_2['Cluster_id'])]

# load cluster data
CI, CII = cluster_df_list[0], cluster_df_list[1]
CII_names = cluster_name_list[1]
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# loop through the 10 CII clsuters
for n, cluster in enumerate(CII_names[:]):

    # read in the isochrones for the two cataloges
    iso_CI = pd.read_csv(empirical_iso_path + "{}_G_BPRP_nboot_1000_cat_1.csv".format(cluster))
    iso_CII = pd.read_csv(empirical_iso_path + "{}_C2_G_BPRP_nboot_1000_cat_2.csv".format(cluster))

    # Create a class object for each cluster
    OC_1 = star_cluster(cluster, CI)
    OC_2 = star_cluster(cluster, CII)

    # Create the CMD that should be used for the isochrone extraction
    OC_1.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])
    OC_2.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])

    # Create a comparison figure for each cluster
    comp_fig = plt.figure(figsize=(4, 6))  # not paper form
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
        comp_fig.savefig(output_path + "{}_CI_vs_CII.pdf".format(cluster), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------