import my_utility

import seaborn as sns
import matplotlib.pyplot as plt

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list

output_path = my_utility.set_output_path()

HP_file = "data/Hyperparameters/Archive_real.csv"
my_utility.setup_HP(HP_file)

# 0.3 Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)
sns.set_style("darkgrid")
save_plot = False

# ----------------------------------------------------------------------------------------------------------------------
examples = ["NGC_2422", "beta Sco"]

fig1 = plt.figure(figsize=(6, 3), layout="constrained")

ax11 = plt.subplot2grid((1, 3), (0, 0))
ax12 = plt.subplot2grid((1, 3), (0, 1))
ax13 = plt.subplot2grid((1, 3), (0, 2))

axes1 = [ax11, ax12, ax13]

fig2 = plt.figure(figsize=(6, 3), layout="constrained")

ax21 = plt.subplot2grid((1, 3), (0, 0))
ax22 = plt.subplot2grid((1, 3), (0, 1))
ax23 = plt.subplot2grid((1, 3), (0, 2))

axes2 = [ax21, ax22, ax23]

h = 0
for cluster, axes, fig in zip(examples, [axes1, axes2], [fig1, fig2]):
    print(cluster)

    # 1. Create a class object for each cluster
    OC = star_cluster(cluster, Archive_df)

    # 2. Create the CMD that should be used for the isochrone extraction
    OC.create_CMD(CMD_params=["Gmag", "BPmag", "Gmag"])

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file, True)
    except IndexError:
        print(OC.name)
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 100
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs)  # output_loc="data/Isochrones/Empirical/"

    # *4.Plot the bootstrapped isochrones
    isochrone_store = np.empty(shape=(len(OC.PCA_XY[:, 0]), 4, n_boot))
    tic = time.time()
    Parallel(n_jobs=10, require="sharedmem")(
        delayed(OC.resample_curves)(idx, output=isochrone_store, kwargs=kwargs) for idx in
        range(n_boot))
    toc = time.time()
    print(toc - tic, "s parallel")

    x_d, y_d, kwargs_CMD = CMD_density_design(OC.CMD, cluster_obj=OC, density_plot=False)
    for i in range(n_boot):
        if cluster == "NGC_2422":
            axes[h + 1].scatter(x_d, y_d, **kwargs_CMD, label="{}".format(OC.name.replace("_", " ")))
        else:
            axes[h + 1].scatter(x_d, y_d, **kwargs_CMD, label=r"$\beta$ {}".format(OC.name.split(" ", maxsplit=2)[1]))

        axes[h + 1].plot(isochrone_store[:, 2, i], isochrone_store[:, 3, i], color="orange", lw=0.5, alpha=0.3,
                         label="resamplings")
    ymin, ymax = axes[h + 1].get_ylim()
    axes[h + 1].set_ylim(ymax, ymin)
    xmin, xmax = axes[h + 1].get_xlim()
    axes[h + 1].set_xlim(xmin, xmax)
    axes[h + 1].set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$")
    axes[h + 1].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")

    px_d, py_d, kwargs_PCA = CMD_density_design(OC.PCA_XY, cluster_obj=OC, density_plot=False)
    for i in range(n_boot):
        axes[h].scatter(px_d, py_d, **kwargs_PCA)
        axes[h].plot(isochrone_store[:, 0, i], isochrone_store[:, 1, i], color="orange", lw=0.5, alpha=0.3)
    # pcmin, pcmax = axes[h].get_ylim()
    pcmax, pcmin = 1.5, -0.4
    axes[h].set_ylim(pcmax, pcmin)
    # axes[h].set_xlim(-4, 10)
    axes[h].set_ylabel("PCA Y")
    axes[h].set_xlabel("PCA X")

    axes[h + 2].scatter(x_d, y_d, **kwargs_CMD)
    axes[h + 2].plot(result_df["l_x"], result_df["l_y"], color="grey")
    axes[h + 2].plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
    axes[h + 2].plot(result_df["u_x"], result_df["u_y"], color="grey", label="5./95. perc")
    axes[h + 2].set_ylim(ymax, ymin)
    axes[h + 2].set_xlim(xmin, xmax)
    axes[h + 2].set_yticklabels([])
    axes[h + 2].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")

    handles, labels = [], []
    for ax in axes[1:]:
        handles_, labels_ = ax.get_legend_handles_labels()
        handles += handles_
        labels += labels_
    axes[h].legend(handles=handles[-4:], labels=labels[-4:], loc='lower left')

    if save_plot:
        fig.savefig(output_path + "bootstrapping_{0}_single.pdf".format(OC.name), dpi=500)

fig1.show()
fig2.show()
