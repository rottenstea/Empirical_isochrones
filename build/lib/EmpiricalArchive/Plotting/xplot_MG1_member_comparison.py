import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list, WD_filter
from EmpiricalArchive.Extraction.Empirical_iso_reader import build_empirical_df

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()

# set other paths
data_path = "//"
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"
reference_ages = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Reference_ages_new.csv")

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Meingast 1
general_kwargs = dict(age_file=reference_ages, col_names=["ref_age", "m_y"], name_split="_G")
BPRP_MG1 = build_empirical_df(filename_key="Meingast_1_G_BPRP_nboot_1000", csv_folder=empirical_iso_path,
                              **general_kwargs)
BPG_MG1 = build_empirical_df(filename_key="Meingast_1_G_BPG_nboot_1000", csv_folder=empirical_iso_path,
                             **general_kwargs)
GRP_MG1 = build_empirical_df(filename_key="Meingast_1_G_GRP_nboot_1000", csv_folder=empirical_iso_path,
                             **general_kwargs)

# Meingast 1 old data
BPRP_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_BPRP_nboot_1000",
                               csv_folder=empirical_iso_path + "/MG 1 ESS II/", **general_kwargs)
BPG_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_BPG_nboot_1000",
                              csv_folder=empirical_iso_path + "/MG 1 ESS II/", **general_kwargs)
GRP_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_GRP_nboot_1000",
                              csv_folder=empirical_iso_path + "/MG 1 ESS II/", **general_kwargs)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# MG IV and II Comparison

# Cluster object 1
MG1_sin_WD = WD_filter(cluster_df_list[5], cols=["Gmag", "BPmag", "RPmag"])
OC1 = star_cluster("Meingast_1", MG1_sin_WD)
OC1.create_CMD(["Gmag", "BPmag", "RPmag"])
OC1x, OC1y, OC1_kwargs = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=False)
OC1_kwargs["s"] = 20

# Cluster object 2
OC2 = star_cluster("Meingast_1_ESSII", cluster_df_list[6])
OC2.create_CMD(["Gmag", "BPmag", "RPmag"])
OC2x, OC2y, OC2_kwargs = CMD_density_design(OC2.CMD, cluster_obj=OC2, from_RBG=[0.678, 0.925, 0.416],
                                            to_RBG=[0.098, 0.318, 0.380], marker="^", s=20, density_plot=False)
OC2_kwargs["s"] = 20

h = plt.figure(figsize=(3.54399, 4.4))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}_{\mathrm{RP}}$")
ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$")
plt.scatter(OC1x, OC1y, label="revised selection", **OC1_kwargs)
plt.scatter(OC2x, OC2y, label="original selection", **OC2_kwargs)
plt.plot(BPRP_MG1["m_x"], BPRP_MG1["m_y"], color="firebrick", label="Isochrone (revised)")
plt.plot(BPRP_MGII["m_x"], BPRP_MGII["m_y"], color="darkorange", label="Isochrone (original)")
plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title("")
plt.subplots_adjust(top=0.99, bottom=0.097, right=0.99, left=0.128)

plt.show()
if save_plot:
    h.savefig(output_path + "Selection_comparison_MG1.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------