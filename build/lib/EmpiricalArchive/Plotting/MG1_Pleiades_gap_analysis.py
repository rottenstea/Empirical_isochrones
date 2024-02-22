import matplotlib.pyplot as plt
import seaborn as sns

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list
from EmpiricalArchive.Extraction.Empirical_iso_reader import merged_BPRP

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
# read in different PARSEC isochrones
high_Av = pd.read_csv(
    "//data/MG1-verification/Pleiades_110Myr_Av018_Z0015_0017.csv")
high_Av["BP-RP"] = high_Av["G_BPmag"] - high_Av["G_RPmag"]

no_Av = pd.read_csv("//data/MG1-verification/PARSEC_110Myr_zvar.csv")
no_Av["BP-RP"] = no_Av["G_BPmag"] - no_Av["G_RPmag"]
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# start and stop indices
aa = 2
bb = 200

# metallicity variation
Z_var = [0.015, 0.016, 0.017]

# colors
palette_1 = ['#1b9e77', ' ', '#d95f02']
palette_2 = ['#7570b3', ' ', '#a6761d']
palette_3 = ["#e7298a", "#7fc97f", "#e6ab02"]

# Paper figure: Zoom-in of lower MS
fig2 = plt.figure(figsize=(3.54399, 3.2))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))
plt.subplots_adjust(left=0.12, right=0.99, top=.81, bottom=.132, wspace=.05)

# for each Z value plot the two Av versions in different linestyles
for id1, z in enumerate(Z_var):

    # leave out second variation for the paper (too many lines)
    if id1 in [0, 2]:
        rel_df2 = no_Av[no_Av["Zini"] == z]
        ax1.plot(rel_df2["BP-RP"].iloc[aa:bb], rel_df2["Gmag"].iloc[aa:bb], color=palette_1[id1], ls="solid",
                 label="Z = {}".format(z))

for id2, z in enumerate(Z_var):
    if id2 in [0, 2]:
        rel_df1 = high_Av[high_Av["Zini"] == z]
        ax1.plot(rel_df1["BP-RP"].iloc[aa:bb], rel_df1["Gmag"].iloc[aa:bb], color=palette_2[id2], ls="--",
                 label="Z = {}, $A_V$ = 0.18".format(z))

ax1.set_xlim(1.5, 4.5)
ax1.set_ylim(14., 8)
ax1.legend(loc="upper center", bbox_to_anchor=(1., 1.3), ncol=2)
ax1.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1)
ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)")

# Define the star clusters
OC1 = star_cluster("Melotte_22", cluster_df_list[1], catalog_mode=True)
OC2 = star_cluster("Blanco_1", cluster_df_list[1], catalog_mode=True)
OC3 = star_cluster("Meingast_1", cluster_df_list[5], catalog_mode=False)

# load isochrone results for the three clusters and plot them
MG_comp = merged_BPRP[merged_BPRP["Cluster_id"].isin(["Meingast_1", "Melotte_22", "Blanco_1"])]

# uncertainty bounds
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_y"],
         color=palette_3[0], lw=1, ls="--", alpha=0.75)
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_y"],
         color=palette_3[0], lw=1, ls="--", alpha=0.75)

ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_y"],
         color=palette_3[1], lw=1, ls="--", alpha=0.75)
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_y"],
         color=palette_3[1], lw=1, ls="--", alpha=0.75)

ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_y"],
         color=palette_3[2], lw=1, ls="--", alpha=0.75)
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_y"],
         color=palette_3[2], lw=1, ls="--", alpha=0.75)

# empirical isochrones
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_y"],
         color=palette_3[0], label="Pleiades")
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_y"],
         color=palette_3[1], label=OC2.name.replace("_", " "))
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_y"],
         color=palette_3[2], label=OC3.name.replace("_", " "))
ax2.set_xlim(1.5, 4.5)
ax2.set_ylim(14., 8)
ax2.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)")
ax2.set_yticklabels([])

plt.show()
if save_plot:
    fig2.savefig(output_path + "Metallicity_Av_effects_110Myr_Comp.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Av vs. no Av comparison plot
fig = plt.figure(figsize=(8, 5.5))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))
plt.subplots_adjust(left=0.1, right=0.99)

# For each Z value plot the two Av values in different linestyles
for z in Z_var:
    rel_df1 = high_Av[high_Av["Zini"] == z]
    rel_df2 = no_Av[no_Av["Zini"] == z]
    ax1.plot(rel_df1["BP-RP"].iloc[aa:bb], rel_df1["Gmag"].iloc[aa:bb], label="Z = {}".format(z))
    ax2.plot(rel_df2["BP-RP"].iloc[aa:bb], rel_df2["Gmag"].iloc[aa:bb], ls="--", label="Z = {}".format(z))

ax1.set_title("highest Av (0.18)")
ax1.set_ylim(14.2, -5)
ax1.set_xlim(-0.5, 5)
ax1.legend(loc="upper right")

ax2.set_title("no Av")
ax2.set_ylim(14.2, -5)
ax2.set_xlim(-0.5, 5)
ax2.legend(loc="upper right")

ax1.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1)
ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1)
ax2.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1)

plt.suptitle("110 Myr PARSEC isochrones")
plt.show()
if save_plot:
    fig.savefig(output_path + "Metallicity_Av_effects_110Myr.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
