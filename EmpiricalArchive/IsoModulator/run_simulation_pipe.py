from Simulation_functions import *
from itertools import product
import seaborn as sns

from EmpiricalArchive.Extraction.pre_processing import cluster_df_list
from EmpiricalArchive.Extraction.Classfile import star_cluster

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.My_tools.plotting_essentials import CMD_density_design

from scipy.integrate import simps

# set paths
output_path = my_utility.set_output_path(
    main_path="/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/")
mastertable_path = "/Users/alena/PycharmProjects/PaperI/EmpiricalArchive/data/Isochrones/Mastertable_Archive.csv"
results_path = "/Users/alena/PycharmProjects/PaperI/EmpiricalArchive/data/Isochrones/Simulations/"

# 0.2 HP file check
HP_file = "/Users/alena/PycharmProjects/PaperI/EmpiricalArchive/data/Hyperparameters/Simulations_1.csv"
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

areas = []
for i, row in enumerate(combinations[40:41]):
    print(row)
    cmd_data = CMD1.simulate(row)
    fig, axes = CMD1.plot_verification(row)

    # Set the background color of the figure
    fig.patch.set_facecolor('black')

    for ax, caption in zip(axes, ["original", r"$\Delta$ parallax", r"$f_{\mathrm{binary}}$", r"$f_{\mathrm{field}}$", "extinction"]):

    # Set the axis background color
        ax.set_facecolor('black')

        # Set the color of the axis labels and tick marks
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Set the color of the spines (axes lines)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

        ax.set_ylim(15, -2)
        ax.set_title(caption, color ="white")

        legend = ax.legend(loc="best", edgecolor="white", facecolor="black")  # without CG
        for text in legend.get_texts():
            text.set_color('white')

    axes[0].set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1, color="white")
    axes[3].set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1, color="white")

    axes[3].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1, color="white")
    axes[4].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1, color="white")

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top = 0.95, hspace=0.3, wspace = 0.3)

    fig.show()
    #fig.savefig(output_path+"IsoModulator_Step1.png", dpi = 500)


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

    # 5. Plot the result
    '''
    fig = CMD_density_design(OC.CMD, cluster_obj=OC)

    # plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="new")
    # plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")
    plt.plot(CMD1.cax, CMD1.abs_G, color="orange", label="old")

    plt.show()
    if save_plot:
        fig.savefig(output_path + f"{OC.name}_{i}_bprp.pdf", dpi=600)
    '''
    # 5. Plot the result
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))  # without CG

    plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="Pleiades")
    # plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], color="orange", lw=2.5, label="New")
    plt.plot(CMD1.cax, CMD1.abs_G, color="magenta", label="old")

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

    plt.title("New emp. isochrone", color="white", y=1.07)
    plt.subplots_adjust(left=0.18, right=0.98)

    legend = ax.legend(loc="best", edgecolor="white", facecolor="black")  # without CG
    for text in legend.get_texts():
        text.set_color('white')
    plt.show()

    if save_plot:
        fig.savefig(output_path + "New_Isochrones_Pleiades.png", dpi=600)

    # Interpolate the second curve onto the x values of the first curve
    y2_interp = np.interp(result_df["m_x"], CMD1.cax, CMD1.abs_G)

    # Calculate the absolute difference between the two curves
    # difference = np.abs(result_df["m_y"] - y2_interp)
    euclidean_distances = np.sqrt((result_df["m_y"] - y2_interp) ** 2)

    # Calculate the area between the curves using the trapezoidal rule
    area_between_curves = simps(euclidean_distances, result_df["m_x"])
    areas.append(area_between_curves)

    print("Area between curves:", area_between_curves)

print("Routine executed sucessfully.")
