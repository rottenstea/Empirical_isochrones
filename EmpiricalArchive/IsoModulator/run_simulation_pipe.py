from Simulation_functions import *
from itertools import product

from EmpiricalArchive.Extraction.pre_processing import cluster_df_list
from EmpiricalArchive.Extraction.Classfile import star_cluster
from EmpiricalArchive.My_tools import my_utility

from comparison_function import compute_NN_distance

# set paths
output_path = my_utility.set_output_path(
    main_path="/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/")
mastertable_path = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Isochrones/Mastertable_Archive.csv"
results_path = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Isochrones/Simulations"

# 0.2 HP file check
HP_file = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Hyperparameters/Simulations_1.csv"
my_utility.setup_HP(HP_file)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=HP_file)

# 0.5 Standard plot settings
# sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 18

save_plot = True

# define uncertainty intervals
f_plx = [0.01, 0.1]
f_bin = [0.2, 0.3, 0.5]
f_cont = [0.05, 0.5, 0.9]
extinct = [0.1, 0.4, 0.789]

# make the grid
combinations = np.array(list(product(f_plx, f_bin, extinct, f_cont)))
# print(combinations.shape)

# define clusters
clusters = ["delta Sco", "Melotte_22", "NGC_2632"]

# load and filter isochrone table and cluster_data_table
mastertable = pd.read_csv(mastertable_path)
filtered_df = mastertable[mastertable["Cluster"].isin(clusters)]
Archive_df = pd.concat(cluster_df_list, axis=0)

CMD1 = simulated_CMD(cluster_name=clusters[1], isochrone_df=filtered_df, cluster_data_df=Archive_df)

# set CMD type
CMD1.set_CMD_type(1)

combination_dfs = []
overall_distances = []

for i, row in enumerate(combinations[:]):
    print(row)
    cmd_data = CMD1.simulate(row)

    OC = star_cluster(name=clusters[1], catalog=cmd_data, dataset_id=i)
    OC.create_CMD_quick_n_dirty(CMD_params=["Gmag", "BP-RP"], no_errors=True)

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
    combination_dfs.append(result_df)

    # 5. Interpolate the new and old isochrones and calculate the distances between the curves, sum up along the curve
    distance = compute_NN_distance(result_df, CMD1.cax, CMD1.abs_G)
    overall_distances.append(round(distance, 2))

# 6. print the results in the plots
for k, df in enumerate(combination_dfs[:]):

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))

    plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="Pleiades")
    plt.plot(df["m_x"], df["m_y"], color="orange", lw=2.5, label="New")
    plt.plot(CMD1.cax, CMD1.abs_G, color="magenta", label="old")

    # Set the figure to dark mode
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # axes
    ax.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1, color="white")
    ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1, color="white")
    ax.set_ylim(15, -2)

    plt.title(f"Difference = {overall_distances[k]} \n {combinations[k]}", color="white", y=1.01)
    plt.subplots_adjust(left=0.18, right=0.98)

    legend = ax.legend(loc="best", edgecolor="white", facecolor="black")  # without CG
    for text in legend.get_texts():
        text.set_color('white')
    plt.show()

    if save_plot:
        fig.savefig(output_path + f"New_isochrones_Pleiades_combo_{k}.png", dpi=300)

print("Routine executed sucessfully.")