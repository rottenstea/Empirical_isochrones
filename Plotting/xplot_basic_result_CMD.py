import matplotlib.pyplot as plt
import seaborn as sns

from My_tools import my_utility
from Extraction.Classfile import *
from Extraction.pre_processing import cluster_df_list, cluster_name_list
from Extraction.Empirical_iso_reader import merged_BPRP

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
# Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

# Collect some example cluster results
example_comp = merged_BPRP[merged_BPRP["Cluster_id"].isin(["Stock_2", "RSG_8", "Ruprecht_147", "NGC_3532"])]

# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# Create a cluster object instance for each of those and plot the CMDs
for n, cluster in enumerate(Archive_clusters):

    if cluster in ["Stock_2", "RSG_8", "Ruprecht_147", "NGC_3532"]:

        df = Archive_df[Archive_df["Cluster_id"] == cluster]
        OC = star_cluster(cluster, df)
        OC.create_CMD()
        CMD_density_design(OC.CMD, cluster_obj=OC, density_plot=False)

        # CMD plot
        basic_CMD, ax = plt.subplots(1, 1, figsize=(3.5, 5))
        plt.subplots_adjust(left=0.17, bottom=0.12, top=0.99, right=0.975, wspace=0.0)

        # Scatter the data
        ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
        ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
        ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$", labelpad=1)

        # Empirical isochrone + bounds
        result_df = example_comp[example_comp["Cluster_id"] == OC.name]
        plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
        plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
        plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[1], ylim[0])
        ax.text(0.2, 9.5, s=OC.name.replace("_", " "))

        basic_CMD.show()
        if save_plot:
            basic_CMD.savefig(output_path + "CMD_{}.pdf".format(OC.name), dpi=500)
# ----------------------------------------------------------------------------------------------------------------------
