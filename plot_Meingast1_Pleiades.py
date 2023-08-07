import my_utility
from Classfile import *
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processing import cluster_df_list, WD_filter

from Empirical_iso_reader import build_empirical_df

# paths
output_path = my_utility.set_output_path()
data_path = "/Users/alena/PycharmProjects/PaperI/"
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"
reference_ages = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Reference_ages_new.csv")

HP = data_path + "data/Hyperparameters/Meingast_1_GRP.csv"
my_utility.setup_HP(HP)

# Meingast 1 files
general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                      name_split="_G")
BPRP_MG1 = build_empirical_df(filename_key="Meingast_1_G_BPRP_nboot_1000", **general_kwargs)
BPG_MG1 = build_empirical_df(filename_key="Meingast_1_G_BPG_nboot_1000", **general_kwargs)
GRP_MG1 = build_empirical_df(filename_key="Meingast_1_G_GRP_nboot_1000", **general_kwargs)


BPRP_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_BPRP_nboot_1000", **general_kwargs)
BPG_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_BPG_nboot_1000", **general_kwargs)
GRP_MGII = build_empirical_df(filename_key="Meingast_1_ESSII_G_GRP_nboot_1000", **general_kwargs)

# merged_Pleiades_I = build_empirical_df(filename_key="Melotte_22_G_BPRP_nboot_1000", filename_exclude="cat_2",
# **general_kwargs)
BPRP_Pleiades_II = build_empirical_df(filename_key="Melotte_22_C2_G_BPRP_nboot_1000", filename_exclude="cat_1",
                                      **general_kwargs)
BPG_Pleiades_II = build_empirical_df(filename_key="Melotte_22_C2_G_BPG_nboot_1000", filename_exclude="cat_1",
                                     **general_kwargs)
GRP_Pleiades_II = build_empirical_df(filename_key="Melotte_22_C2_G_GRP_nboot_1000", filename_exclude="cat_1",
                                     **general_kwargs)

# plotting settings
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

kwargs = dict(grid=None, HP_file=HP)
save_plot = True


# MG IV and II Comparison
MG1_sin_WD = WD_filter(cluster_df_list[5], cols=["Gmag", "BPmag", "RPmag"])
OC1 = star_cluster("Meingast_1", MG1_sin_WD)
OC1.create_CMD(["Gmag", "BPmag", "RPmag"])
OC1x, OC1y, OC1_kwargs = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=False)
OC1_kwargs["s"] = 20

OC2 = star_cluster("Meingast_1_ESSII", cluster_df_list[6])
OC2.create_CMD(["Gmag", "BPmag", "RPmag"])
OC2x, OC2y, OC2_kwargs = CMD_density_design(OC2.CMD, cluster_obj=OC2, from_RBG=[0.678, 0.925, 0.416],
                                            to_RBG=[0.098, 0.318, 0.380], marker="^", s=20, density_plot=False)


OC1_kwargs["s"] = 20

h = plt.figure(figsize=(3.54399,4.4))
ax = plt.subplot2grid((1,1), (0,0))
ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}_{\mathrm{RP}}$")
ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$")
plt.scatter(OC1x,OC1y, label = "revised selection", **OC1_kwargs)
plt.scatter(OC2x,OC2y, label = "original selection", **OC2_kwargs)
plt.plot(BPRP_MG1["m_x"], BPRP_MG1["m_y"], color="firebrick", label="Isochrone (revised)")
plt.plot(BPRP_MGII["m_x"], BPRP_MGII["m_y"], color="darkorange", label="Isochrone (original)")
plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title("")
plt.subplots_adjust(top=0.99,bottom=0.097,right=0.99, left = 0.128)

plt.show()
if save_plot:
    h.savefig(output_path+"Selection_comparison_MG1.pdf", dpi=600)

'''

f = plt.figure(figsize=(8, 4))

ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1), sharey=ax1)
ax3 = plt.subplot2grid((1, 3), (0, 2), sharey=ax1)

filters = [["Gmag", "BPmag", "RPmag"], ["Gmag", "BPmag", "Gmag"], ["Gmag", "Gmag", "RPmag"]]
axes = [ax1, ax2, ax3]
isos_MG1 = [BPRP_MG1, BPG_MG1, GRP_MG1]
isos_MGII = [BPRP_MGII, BPG_MGII, GRP_MGII]

isos_P = [BPRP_Pleiades_II, BPG_Pleiades_II, GRP_Pleiades_II]

for ax, cmd, iM, iP in zip(axes, filters, isos_MG1, isos_P):
    # Meingast 1 cluster
    MG1_sin_WD = WD_filter(cluster_df_list[5], cols=cmd)
    MG1 = MG1_sin_WD[MG1_sin_WD["stability"] > 50]
    OC1 = star_cluster("Meingast_1", MG1)
    OC1.create_CMD(cmd)
    OC1x, OC1y, OC1_kwargs = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=False)
    h = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=True)
    plt.plot(iM["m_x"], iM["m_y"], color="firebrick", label="Meingast 1")
    h.show()

    if save_plot:
        h.savefig(output_path+"{0}_{1}_iso.pdf".format(OC1.name, OC1.CMD_specs["short"]),dpi=500)

    # Pleiades
    OC2_II = star_cluster("Melotte_22", cluster_df_list[1])
    OC2_II.create_CMD(cmd)
    OC2x, OC2y, OC2_kwargs = CMD_density_design(OC2_II.CMD, cluster_obj=OC2_II, from_RBG=[0.74, 0.74, 0.74],
                                                to_RBG=[0.0, 0.0, 0.0], marker="^", s=20, density_plot=False)
    j = CMD_density_design(OC2_II.CMD, cluster_obj=OC2_II, from_RBG=[0.74, 0.74, 0.74], to_RBG=[0.0, 0.0, 0.0],
                           marker="^", density_plot=True)
    plt.plot(iP["m_x"], iP["m_y"], color="darkorange", label="Pleiades (CII)")
    j.show()
    if save_plot:
        j.savefig(output_path+"{0}_{1}_iso.pdf".format(OC2_II.name, OC2_II.CMD_specs["short"]),dpi=500)

    ax.scatter(OC1x,OC1y,**OC1_kwargs, label = "MG1 data")
    ax.scatter(OC2x,OC2y,**OC2_kwargs, label = "Pleiades data")
    ax.plot(iP["m_x"], iP["m_y"], color="darkorange", label="Pleiades (CII)")
    ax.plot(iM["m_x"], iM["m_y"], color="firebrick", label="Meingast 1")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymax, ymin)

ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}_{\mathrm{RP}}$")
ax1.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$")
ax2.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}$")
ax3.set_xlabel(r"$\mathrm{G} -\mathrm{G}_{\mathrm{RP}}$")

ax3.legend(loc="center right", bbox_to_anchor=(1.8,0.5))

ax2.set_yticklabels([])
ax3.set_yticklabels([])
plt.subplots_adjust(left=0.05,right=0.81, top=0.98,bottom=0.11, wspace=0.1)
f.show()

if save_plot:
    f.savefig(output_path + "Comparison_{0}_{1}.pdf".format(OC1.name, OC2_II.name), dpi=500)
'''
'''
# 3. Do some initial HP tuning if necessary
# try:
#     params = OC1.SVR_read_from_file(HP, True)
# except IndexError:
#     print(OC1.name)
curve, isochrone = OC1.curve_extraction(OC1.PCA_XY, **kwargs)

# 4. Create the robust isochrone and uncertainty border from bootstrapped curves
n_boot = 1000
result_df = OC1.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs,
                                        parallel_jobs=10 , output_loc="data/Isochrones/Empirical/")

fig = CMD_density_design(OC1.CMD, cluster_obj=OC1)

plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")
plt.show()

# OC1x, OC1y, OC1_kwargs = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=False)
# h = CMD_density_design(OC1.CMD, cluster_obj=OC1, density_plot=True)
# plt.plot(GRP_MG1["m_x"], GRP_MG1["m_y"], color="red", label="Meingast 1")
# h.show()

'''
