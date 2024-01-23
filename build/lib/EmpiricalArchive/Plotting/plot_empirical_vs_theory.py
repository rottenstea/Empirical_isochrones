import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import case_study_names, case_study_dfs

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()

# load and set HP paths and files
data_path = "//"

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False

colors = ["red", "darkorange"]
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Load theoretical isochrones
# ----------------------------
# BHAC15 from the website
bhac15_df = pd.read_csv(data_path + "data/Isochrones/BHAC15/baraffe15.csv")
# PARSEC from Nuria
parsec_df_1 = pd.read_csv(data_path +
                          "data/Isochrones/PARSEC_isochrones/PARSEC+COLIBRI_30Myr_GDR2-Evans_2MASS_ps1.csv")
# Current PARSEC download with OBC
parsec_df = pd.read_csv(data_path +
                        "data/Isochrones/PARSEC_isochrones/2MASS_GAIA_PS_30_Myr.csv")
# BT-Settl (also from Nuria)
btsettl_df = pd.read_csv(data_path +
                         "data/Isochrones/PARSEC_isochrones/Nuria_clusters/BTSettl_30Myr_GDR2_ps1_2mass.csv")

# Load empirical isochrones
# --------------------------
archive_iso = pd.read_csv(data_path + "data/Isochrones/Empirical/DANCe_system/IC_4665_i_iK_nboot_1000_cat_None.csv")
empirical_Nuria = pd.read_csv(data_path + "data/Isochrones/Empirical_Nuria/empirical_sequence_izyJHK.csv")

# IC 4665 data
IC4665_cluster, IC4665_df = case_study_names[1], case_study_dfs[1]
IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13)]
IC4665_age = [0.025, 0.03, 0.04]
isos_IC4665 = [bhac15_df[bhac15_df["Age_p1"] == i] for i in IC4665_age]

# Define cluster object and CMD
OC = star_cluster("IC_4665", IC4665_filtered_df, False)
OC.create_CMD(CMD_params=["imag", "imag", "Kmag"])
OCx, OCy, OC_kwargs = CMD_density_design(OC.CMD, cluster_obj=OC, density_plot=False,
                                         from_RBG=[0.74, 0.74, 0.74], to_RBG=[0.0, 0.0, 0.0])

OC.kwargs_CMD["s"] = 20
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
palette_3 = ["#e7298a", "#7fc97f", "#e6ab02"]

fig = plt.figure(figsize=(3.54399, 3.2))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))
plt.subplots_adjust(left=0.115, right=0.99, top=.93, bottom=.117, wspace=.05)

# Subplot 1: Theoretical isochrones
ax1.scatter(OCx, OCy + 5 * np.log10(1000 / OC.data.plx[0]) - 5, **OC_kwargs)

for j, isos in enumerate(isos_IC4665[:]):

    # only use the 30 Myr isochrones for BHAC15
    if j == 1:
        ax1.plot(isos["i_p1"] - isos["Mk"] + 0.39644, isos["i_p1"] + 5 * np.log10(1000 / OC.data.plx[0]) - 5 + 0.4526,
                 color=palette_3[0], label="BHAC15")

ax1.plot(parsec_df["iP1mag"][3:] - parsec_df["Ksmag"][3:] + 0.39644,
         parsec_df["iP1mag"][3:] + 5 * np.log10(1000 / OC.data.plx[0]) - 5 + 0.4526, color=palette_3[1],
         label="PARSEC")
# plt.plot(parsec_df_1["iP1mag"] - parsec_df_1["Ksmag"] + 0.39644,
#         parsec_df_1["iP1mag"] + 5 * np.log10(1000 / OC.data.plx[0]) - 5 + 0.4526, color="firebrick",
#         label="PARSEC 30 Myr NURIA")
ax1.plot(btsettl_df["i_p1"] - btsettl_df["K"] + 0.39644,
         btsettl_df["i_p1"] + 5 * np.log10(1000 / OC.data.plx[0]) - 5 + 0.4526, color=palette_3[2],
         label="BT-Settl")

ax1.set_title("Theoretical", fontsize=10)
ax1.set_ylim(25.5, 11)
ax1.set_xlim(1.75, 7)
ax1.set_ylabel(r"$\mathrm{m}_{\mathrm{i}}$ (mag)", labelpad=2)
ax1.set_xlabel(r"$\mathrm{i} - \mathrm{K}$ (mag)", labelpad=2, x=1)
ax1.legend(loc="upper right")

# Subplot 2: Empirical
ax2.scatter(OCx, OCy + 5 * np.log10(1000 / OC.data.plx[0]) - 5, **OC_kwargs)
ax2.plot(empirical_Nuria["i"] - empirical_Nuria["K"], empirical_Nuria["i"], color="darkorange", label="DANCe")
ax2.plot(archive_iso["m_x"], archive_iso["m_y"] + 5 * np.log10(1000 / OC.data.plx[0]) - 5, color="red",
         label="This work")
ax2.set_title("Empirical", fontsize=10)
ax2.legend(loc="upper right")
ax2.set_ylim(25.5, 11)
ax2.set_xlim(1.75, 7)
ax2.set_yticklabels([])

plt.show()
if save_plot:
    fig.savefig(output_path + "{}_comparison.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
