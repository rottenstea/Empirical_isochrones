import os
from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list

# 0.1 Set the correct output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

# 0.2 HP file check
HP_file = "data/Hyperparameters/CatalogII.csv"
try:
    pd.read_csv(HP_file)
except FileNotFoundError:
    with open(HP_file, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")


# 0.3 Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=HP_file)
sns.set_style("darkgrid")
save_plot = True

# ----------------------------------------------------------------------------------------------------------------------

# closer look at CII
CII_df = cluster_df_list[1]
CII_clusters = cluster_name_list[1]
for n, cluster in enumerate(CII_clusters[:]):

    # 1. Create a class object for each cluster
    OC = star_cluster(cluster, CII_df)

    # 2. Create the CMD that should be used for the isochrone extraction
    OC.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file)
    except IndexError:
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 1000
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, output_loc = "data/Isochrones/", kwargs = kwargs)

    # 5. Plot the result
    fig = CMD_density_design(OC.CMD, cluster_obj=OC)

    plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label ="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], color="red", label = "Isochrone")
    plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label = "95. perc")

    plt.show()
    if save_plot:
        fig.savefig(output_path+"{0}_{1}.pdf".format(OC.name,OC.CMD_specs["short"]), dpi = 500)




#------------------------------------------------------------------

# CODE SNIPPETS FOR SUMMARY PLOT

# Fig, axes = plt.subplots(figsize=(12, 23), nrows=10, ncols=7)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

# collect isos and OCs for later
# iso_array = []
# OCs = []
#
# for i, ax in enumerate(axes.flat[:]):
#     try:
#
#         OC = OCs[i]
#         kr = 0
#
#         sns.set_style("darkgrid")
#
#         cs = ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
#         ax.plot(iso_array[i][kr:, 0], iso_array[i][kr:, 1], color="red")
#
#         # ax.set_title(OC.name.replace("_", " "))
#         # if i in range(0, 70, 7):
#         #     ax.set_ylabel(f"absolute {OC.CMD_specs['axes'][0]}")
#         #
#         # if i in range(63, 70):
#         #     ax.set_xlabel(OC.CMD_specs["axes"][1])
#
#         ymin, ymax = ax.get_ylim()
#         ax.set_ylim(ymax, ymin)
#
#         plt.colorbar(cs, ax=ax)
#
#     except IndexError:
#         pass
#
# plt.subplots_adjust(wspace=0.5, hspace=0.4)
#
# Fig.show()
# Fig.savefig(output_path + "composite_Archive_normweights.pdf", dpi=500)
