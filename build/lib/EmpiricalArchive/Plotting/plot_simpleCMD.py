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

save_plot = True
# ----------------------------------------------------------------------------------------------------------------------
# EXTRACTION ROUTINE
# ----------------------------------------------------------------------------------------------------------------------

OC = star_cluster("Melotte_22", Archive_df, dataset_id=2)

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
n_boot = 1000
result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, output_loc=results_path)

# 5. Plot the result
fig, ax = plt.subplots(1, 1, figsize=(4, 6))  # without CG


plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="Pleiades")
# plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
plt.plot(result_df["m_x"], result_df["m_y"], color="magenta", lw =2.5,label="Isochrone")
# plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

# Set the background color of the figure
fig.patch.set_facecolor('black')

# Set the axis background color
ax.set_facecolor('black')

# Set the color of the axis labels and tick marks
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1, color="white")
ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1, color="white")

# Set the color of the spines (axes lines)
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')

ax.set_ylim(15, -2)

plt.title("Empirical isochrone", color="white", y = 1.07)
plt.subplots_adjust(left=0.18, right = 0.98)


legend = ax.legend(loc="best", edgecolor="white", facecolor="black")  # without CG
for text in legend.get_texts():
    text.set_color('white')
plt.show()

if save_plot:
    fig.savefig(output_path + "Isochrone_Pleiades_magenta.png", dpi=600)
