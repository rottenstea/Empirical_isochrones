import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list, cluster_name_list

""" Blueprint file for the isochrone extraction routine."""

# 0.1 Set the correct output paths
output_path = my_utility.set_output_path()
results_path = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/"

# 0.2 HP file check
HP_file = "//EmpiricalArchive/data/Hyperparameters/Archive_full.csv"
my_utility.setup_HP(HP_file)

# 0.3 Create the archive from all the loaded data files
Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=HP_file)

# 0.5 Standard plot settings
# sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 18

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# EXTRACTION ROUTINE
# ----------------------------------------------------------------------------------------------------------------------

for cluster in Archive_clusters[:]:

    if cluster in ["Melotte_22", "Pleiades"]:

        # 1. Create a class object for each cluster
        OC = star_cluster(cluster, Archive_df, dataset_id=2)

        # 2. Create the CMD that should be used for the isochrone extraction
        OC.create_CMD(CMD_params=["Gmag", "BPmag", "Gmag"])

        # 3. Do some initial HP tuning if necessary
        try:
            params = OC.SVR_read_from_file(HP_file)
        except IndexError:
            print(f"No Hyperparameters were found for {OC.name}.")
            curve, isochrone = OC.curve_extraction(svr_data=OC.PCA_XY, svr_weights=OC.weights,
                                                   svr_predict=OC.PCA_XY[:, 0], **kwargs)

        # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
        n_boot = 100
        result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, output_loc=results_path)

        # 5. Plot the result
        fig = CMD_density_design(OC.CMD, cluster_obj=OC)

        plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
        plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
        plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

        plt.show()
        if save_plot:
            fig.savefig(output_path + f"{OC.name}.pdf", dpi=600)

print("Routine executed sucessfully.")
# ----------------------------------------------------------------------------------------------------------------------
