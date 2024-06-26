import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list, cluster_name_list

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
HP_file = "//data/Hyperparameters/Archive_full.csv"
my_utility.setup_HP(HP_file)
kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
fig1 = plt.figure(figsize=(3.54399, 3.7))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 0.4, 1, 1])
ax11 = plt.subplot(gs[0])
ax12 = plt.subplot(gs[2])
ax13 = plt.subplot(gs[3])
ax21 = plt.subplot(gs[4])
ax22 = plt.subplot(gs[6])
ax23 = plt.subplot(gs[7])
plt.subplots_adjust(top=0.855, right=0.99, bottom=0.107, hspace=0.21, left=0.125, wspace=0.05)

# each row corresponds to one cluster
axes1 = [ax11, ax12, ax13]
axes2 = [ax21, ax22, ax23]

h = 0  # ax indexer
count = [0, 1]  # for correct ax label placement

# Example clusters
examples = ["NGC_2422", "beta Sco"]

# For each cluster, create a CMD and 100 isochroen resamplings + results
for c, cluster, axes in zip(count, examples, [axes1, axes2]):

    print(cluster)
    OC = star_cluster(cluster, Archive_df)
    OC.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])
    x_d, y_d, kwargs_CMD = CMD_density_design(OC.CMD, cluster_obj=OC, density_plot=False)
    px_d, py_d, kwargs_PCA = CMD_density_design(OC.PCA_XY, cluster_obj=OC, density_plot=False)

    # Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file, True)
    except IndexError:
        print(OC.name)
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # results
    n_boot = 100
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs)  # output_loc="data/Isochrones/Empirical/"

    # extra resampling (for visualization) and storing of the isochrones in array
    isochrone_store = np.empty(shape=(len(OC.PCA_XY[:, 0]), 4, n_boot))
    tic = time.time()
    Parallel(n_jobs=10, require="sharedmem")(
        delayed(OC.resample_curves)(idx, output=isochrone_store, kwargs=kwargs) for idx in
        range(n_boot))
    toc = time.time()
    print(toc - tic, "s parallel")

    # Subplot 1
    axes[h].scatter(px_d, py_d, **kwargs_PCA)

    # Write cluster name in lower left corner of the first plot of each row
    axes[h].text(-2.5, 1.4, s=cluster.replace("_", " "), fontsize=10)

    # plot all resampled PCA curves
    for i in range(n_boot):
        axes[h].plot(isochrone_store[:, 0, i], isochrone_store[:, 1, i], color="orange", lw=0.5, alpha=0.3)
    pcmax, pcmin = 1.5, -0.4
    axes[h].set_ylim(pcmax, pcmin)
    axes[h].set_ylabel("PCA Y", labelpad=1)

    # X-axis labels
    if c == 1:
        axes[h].set_xlabel("PCA X")

    # Subplot 2
    axes[h + 1].scatter(x_d, y_d, **kwargs_CMD, label="data")

    # plot all the resampled isochrones
    for i in range(n_boot):
        axes[h + 1].plot(isochrone_store[:, 2, i], isochrone_store[:, 3, i], color="orange", lw=0.5, alpha=0.3,
                         label="resamplings")
        if i == int(n_boot - 1):
            axes[h + 1].scatter(x_d, y_d, **kwargs_CMD, label="data")
    ymin, ymax = axes[h + 1].get_ylim()
    axes[h + 1].set_ylim(ymax, ymin)
    xmin, xmax = axes[h + 1].get_xlim()
    axes[h + 1].set_xlim(xmin, xmax)
    axes[h + 1].set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$ (mag)", labelpad=-1)
    if c == 1:
        axes[h + 1].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", x=1., labelpad=2)

    # Subplot 3
    axes[h + 2].scatter(x_d, y_d, **kwargs_CMD)
    axes[h + 2].plot(result_df["l_x"], result_df["l_y"], color="black", alpha=0.75, lw=1)
    axes[h + 2].plot(result_df["u_x"], result_df["u_y"], color="black", alpha=0.75, label="5p/95p bounds", lw=1)
    axes[h + 2].plot(result_df["m_x"], result_df["m_y"], color="red", label="isochrone")
    axes[h + 2].set_ylim(ymax, ymin)
    axes[h + 2].set_xlim(xmin, xmax)
    axes[h + 2].set_yticklabels([])

    # custom legend
    if cluster == "NGC_2422":
        handles, labels = [], []
        for ax in axes[1:]:
            handles_, labels_ = ax.get_legend_handles_labels()
            handles += handles_
            labels += labels_
        axes[h].legend(handles=handles[-4:], labels=labels[-4:], loc='upper center', ncol=2, bbox_to_anchor=(1.5, 1.45))

fig1.show()
if save_plot:
    fig1.savefig(output_path + "Bootstrapping.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
