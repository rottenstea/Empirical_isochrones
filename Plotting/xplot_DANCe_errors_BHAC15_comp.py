import seaborn as sns
import matplotlib.pyplot as plt

from My_tools import my_utility
from Extraction.Classfile import *
from Extraction.pre_processing import case_study_names, cluster_df_list, case_study_dfs

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()

# Set HP file for the Case studies
HP_file_cs = "/Users/alena/PycharmProjects/PaperI/data/Hyperparameters/DANCe_clusters.csv"
my_utility.setup_HP(HP_file_cs)
kwargs = dict(grid=None, HP_file=HP_file_cs, catalog_mode=False)

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

colors = ["red", "darkorange"]
save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA CRUNCHING
# ----------------------------------------------------------------------------------------------------------------------
# Pleiades
Pleiades_cluster, Pleiades_df = case_study_names[0], case_study_dfs[0]
# filter saturated sources
Pleiades_filtered_df = Pleiades_df[Pleiades_df["imag"] > 13]

# IC 4665
IC4665_cluster, IC4665_df = case_study_names[1], case_study_dfs[1]

# filter saturated sources
IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13)]

# form one large dataframe
N_df = pd.concat([Pleiades_filtered_df, IC4665_filtered_df], axis=0)
# ----------------------------------------------------------------------------------------------------------------------
# ISOCHRONE COMPUTATION
# ----------------------------------------------------------------------------------------------------------------------
# store cluster objects for later
OCs = []
for i, cluster in enumerate(case_study_names[:]):
    OC = star_cluster(cluster, N_df, catalog_mode=False)
    deltas = OC.create_CMD(CMD_params=["imag", "imag", "Kmag"], return_errors=True)

    # Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file_cs, catalog_mode=False)
    except IndexError:
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 100
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, parallel_jobs=10)

    # Save the results as cluster attributes
    setattr(OC, "isochrone", result_df[["m_x", "m_y"]])
    setattr(OC, "upper", result_df[["u_x", "u_y"]])
    setattr(OC, "lower", result_df[["l_x", "l_y"]])

    OCs.append(OC)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# Plot the isochrones and boundaries on top of error-color coded data
cm = plt.cm.get_cmap("crest")

for OC in OCs:
    fig1 = plt.figure(figsize=(4, 6))  # not paper-format

    ax1 = plt.subplot2grid((1, 1), (0, 0))
    sc = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, s=20)
    ax1.plot(OC.lower["l_x"], OC.lower["l_y"], color="grey", label="5. perc")
    ax1.plot(OC.isochrone["m_x"], OC.isochrone["m_y"], color="red", label="Isochrone")
    ax1.plot(OC.upper["u_x"], OC.upper["u_y"], color="grey", label="95. perc")
    plt.colorbar(sc)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])
    ax1.set_xlabel(OC.CMD_specs["axes"][1])
    ax1.set_title(OC.name)

    plt.show()
    if save_plot:
        fig1.savefig(output_path + "{}_isochrone_errormap.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Error matrix plot
for OC in OCs:
    deltas = OC.create_CMD(CMD_params=["imag", "imag", "Kmag"], return_errors=True)

    error_fig = plt.figure()  # not paper-format

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1), sharey=ax1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
    ax4 = plt.subplot2grid((2, 2), (1, 1), sharey=ax3)

    # Subplot 1
    s1 = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[0], cmap=cm, marker=".", s=5)
    ax1.set_title("imag error (cax)")
    plt.colorbar(s1, ax=ax1)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])

    # Subplot 2
    s2 = ax2.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[1], cmap=cm, marker=".", s=5)
    ax2.set_title("Kmag error (cax)")
    plt.colorbar(s2, ax=ax2)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymax, ymin)

    # Subplot 3
    cax_error = np.sqrt(deltas[0] ** 2 + deltas[1] ** 2)
    s3 = ax3.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=cax_error, cmap=cm, marker=".", s=5)
    ax3.set_title("cax errors")
    plt.colorbar(s3, ax=ax3)
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymax, ymin)
    ax3.set_ylabel(OC.CMD_specs["axes"][0])
    ax3.set_xlabel(OC.CMD_specs["axes"][1])

    # Subplot 4
    s4 = ax4.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, marker=".", s=5)
    ax4.set_title("weights")
    plt.colorbar(s4, ax=ax4)
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(ymax, ymin)
    ax4.set_xlabel(OC.CMD_specs["axes"][1])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()
    if save_plot:
        error_fig.savefig(output_path + "{}_isochrone_errormatrix.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# BHAC15 isochrones for GAIA Pleiades data
# ----------------------------------------------------------------------------------------------------------------------
# DATA + EXTRACTION
# ----------------------------------------------------------------------------------------------------------------------
# Set new HP file
HP_CII = "/Users/alena/PycharmProjects/PaperI/data/Hyperparameters/CatalogII.csv"
my_utility.setup_HP(HP_CII)

# Load theoretical isochrones for the Pleiades
bhac15_df = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Isochrones/BHAC15/baraffe15.csv")
Pleiades_age = [0.08, 0.1, 0.12]  # in Gyr
isos_Pleiades = [bhac15_df[bhac15_df["Age_GAIA"] == i] for i in Pleiades_age]

# create cluster object and isochrones
CII_df = cluster_df_list[1]
OC = star_cluster("Melotte_22", CII_df)
OC.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])
try:
    params = OC.SVR_read_from_file(HP_CII, catalog_mode=False)
except IndexError:
    curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)
n_boot = 1000
result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
fig_BPRP = plt.figure(figsize=(5, 6))  # not paper-format
plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)

# plot the different theoretical isochrones
for j, isos in enumerate(isos_Pleiades):
    plt.plot(isos["G_BP"] - isos["G_RP"], isos["G"], label="{} Myr".format(int(Pleiades_age[j] * 10 ** 3)))

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name)
plt.show()
if save_plot:
    fig_BPRP.savefig(output_path + "{}_BPRP_BHAC15.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# same for G-RP CMD
OC.create_CMD(CMD_params=["Gmag", "Gmag", "RPmag"])
try:
    params = OC.SVR_read_from_file(HP_CII, catalog_mode=False)
except IndexError:
    curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)
n_boot = 1000
result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs)

fig_GRP = plt.figure(figsize=(5, 6))  # not paper-format
plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)

for j, isos in enumerate(isos_Pleiades):
    plt.plot(isos["G"] - isos["G_RP"], isos["G"], label="{} Myr".format(int(Pleiades_age[j] * 10 ** 3)))

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name)
plt.show()
if save_plot:
    fig_GRP.savefig(output_path + "{}_GRP_BHAC15.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
