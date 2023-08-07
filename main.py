import my_utility

import seaborn as sns
import matplotlib.pyplot as plt

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list, WD_filter, CIII_clusters_new, CIII_df

# 0.1 Set the correct output paths
output_path = my_utility.set_output_path()

# 0.2 HP file check
HP_file = "data/Hyperparameters/CIII_wo_WD.csv"
my_utility.setup_HP(HP_file)

# 0.3 Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

# CMDspec

CMDspec = ["Gmag", "BPmag", "RPmag"]
# M = cluster_df_list[5]
# M_name = cluster_name_list[5]
# M_sin_WD = WD_filter(M, cols=CMDspec)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
# evals = np.logspace(-2, -1.5, 20)
# Cvals = np.logspace(-1, 2, 20)
# grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals, epsilon=evals)
# kwargs = dict(grid=grid, HP_file=HP_file, catalog_mode=True)
kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)
sns.set_style("darkgrid")
save_plot = True

# ----------------------------------------------------------------------------------------------------------------------


for n, cluster in enumerate(["sigma Sco"]):
    # 1. Create a class object for each cluster
    df = CIII_df[CIII_df["Cluster_id"] == cluster]
    df_WO = WD_filter(df, CMDspec)
    OC = star_cluster(cluster, df_WO)

    # 2. Create the CMD that should be used for the isochrone extraction
    errors = OC.create_CMD(CMD_params=CMDspec, return_errors=True)

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file, True)
    except IndexError:
        print(OC.name)
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 1000
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs,
                                           output_loc="data/Isochrones/White_dwarf_filter_test/")

    # 5. Plot the result
    fig = CMD_density_design(OC.CMD, cluster_obj=OC)

    plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
    plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

    plt.show()
    if save_plot:
        pass
        #fig.savefig(output_path + "{0}_{1}_cat_{2}_no_WD.pdf".format(OC.name, OC.CMD_specs["short"], OC.catalog_id),
         #           dpi=500)

#
    # error fig
    cm = plt.cm.get_cmap("crest")

    fig1 = plt.figure(figsize=(4, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    sc = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=errors[0], cmap=cm, s=20)
    ax1.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    ax1.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
    ax1.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")
    plt.colorbar(sc)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])
    ax1.set_xlabel(OC.CMD_specs["axes"][1])
    ax1.set_title(OC.name)

    fig1.show()
    #if save_plot:
     #   fig1.savefig(output_path+"Delta_c1_{}.png".format(OC.name),dpi=500)