import my_utility

import seaborn as sns
import matplotlib.pyplot as plt

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list

# 0.1 Set the correct output paths
output_path = my_utility.set_output_path()

# 0.2 HP file check
HP_file = "data/Hyperparameters/Archive_real.csv"
my_utility.setup_HP(HP_file)

# 0.3 Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

CIII = cluster_df_list[2]
CIII_names = cluster_name_list[2]

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)
sns.set_style("darkgrid")
save_plot = False

# ----------------------------------------------------------------------------------------------------------------------

for n, cluster in enumerate(CIII_names[:]):
    print(cluster)
    # 1. Create a class object for each cluster
    OC = star_cluster(cluster, CIII)

    # 2. Create the CMD that should be used for the isochrone extraction
    OC.create_CMD(CMD_params=["Gmag", "Gmag", "RPmag"])

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file, True)
    except IndexError:
        print(OC.name)
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 1000
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, output_loc="data/Isochrones/Empirical/")

    # 5. Plot the result
    fig = CMD_density_design(OC.CMD, cluster_obj=OC)

    plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
    plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

    plt.show()
    if save_plot:
        fig.savefig(output_path + "{0}_{1}_cat_{2}.pdf".format(OC.name, OC.CMD_specs["short"], OC.catalog_id), dpi=500)

#