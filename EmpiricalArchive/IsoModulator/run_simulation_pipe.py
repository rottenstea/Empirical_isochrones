from Simulation_functions import *
from itertools import product

from EmpiricalArchive.Extraction.pre_processing import cluster_df_list
from EmpiricalArchive.Extraction.Classfile import star_cluster
from EmpiricalArchive.My_tools import my_utility

from comparison_function import compute_NN_distance, anova_analysis

import seaborn as sns

# set paths
output_path = my_utility.set_output_path(
    main_path="/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/")
mastertable_path = \
    "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Isochrones/Mastertable_Archive.csv"
results_path = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Isochrones/Simulations/Fixed_Extinction/"

# HP file check
HP_file = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Hyperparameters/Simulations_test_Extinctionfix.csv"
my_utility.setup_HP(HP_file)

# Set the kwargs for the parameter grid and HP file and plot specs
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 12
kwargs = dict(grid=None, HP_file=HP_file)
save_plot = True

##########
## GRID ##
##########

# define uncertainty intervals
f_plx = [0.0, 0.05, 0.1, 0.3]
f_bin = [0.1, 0.3, 0.5, 0.8]
extinct = [0.0, 0.2, 0.5, 1]
f_cont = [0.01, 0.25, 0.5, 1]

combinations = np.array(list(product(f_plx, f_bin, extinct, f_cont)))  # make the grid
combinations = np.insert(combinations, 0, [0, 0, 0, 0],
                         axis=0)  # insert 0 values for min calibration of the distance metric


clusters = ["delta Sco", "Blanco_1", "NGC_752"]  # define clusters
# color_palettes = ["Blues"]#, "Oranges"]
palette_3 = ["#e7298a", "#7fc97f", "#e6ab02"]
#################
# SIMULATE CMDS #
#################

mastertable = pd.read_csv(mastertable_path)  # table of all empirical isochrones in the archive
filtered_df = mastertable[mastertable["Cluster"].isin(clusters)]  # filter for the example clusters
Archive_df = pd.concat(cluster_df_list, axis=0)  # cluster data table (for distance approximation)

all_dfs = []
# Loop over clusters
for c_id,c in enumerate(clusters[:]):
    print("-" * 60, f"\n RUN FOR {c} \n", 60 * "-")

    combination_df = pd.DataFrame(data=combinations,
                                  columns=["u_plx", "f_binaries", "extinction", "f_field"])  # define grid
    sim_CMD = simulated_CMD(cluster_name=c, isochrone_df=filtered_df,
                            cluster_data_df=Archive_df)  # simulate CMD from empirical isochrone

    CMD_type_combos = []
    # Loop over CMD types
    for typ in [1, 2, 3]:
        print(f'\n :: Run for CMD type {typ} :: \n')

        sim_CMD.set_CMD_type(typ)  # set CMD type
        combination_dfs = []
        OCs = []

        # loop over grid points
        for i in range(len(combinations))[:]:
            print(f'{c} - CMD {typ} - combi: {i}')

            row = combination_df.iloc[i].values[:4]  # set grid point values
            cmd_data = sim_CMD.simulate(row)  # simulate CMD with the values

            # fig, axes = sim_CMD.plot_verification(row)
            # plt.title(f"{c}, CMD {typ}")
            # fig.show()

            OC = star_cluster(name=c, catalog=cmd_data, dataset_id=i)  # make cluster object from new CMD data
            OC.create_CMD_quick_n_dirty(CMD_params=sim_CMD.cols[::-1], no_errors=True)  # Create CMD

            # Extraction routine
            try:
                params = OC.SVR_read_from_file(HP_file)
            except IndexError:
                print(f"No Hyperparameters were found for {OC.name}.")
                curve, isochrone = OC.curve_extraction(svr_data=OC.PCA_XY, svr_weights=OC.weights,
                                                       svr_predict=OC.PCA_XY[:, 0], **kwargs)

            n_boot = 1000
            result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, output_loc=results_path)
            combination_dfs.append(result_df)

            # Interpolate the new and old isochrones and calculate the NN distance sum
            distance = compute_NN_distance(result_df, sim_CMD.cax, sim_CMD.abs_G)
            combination_df.loc[i, f"CMD_{typ}"] = round(distance, 2)  # store distance in grid dataframe
            OCs.append(OC)  # save cluster object for plotting

        CMD_type_combos.append(combination_dfs)

        # Apply rescaling to the 'value' column
        combination_df[f"scaled_CMD_{typ}"] = combination_df[f"CMD_{typ}"].apply(
            lambda x: (x - combination_df[f"CMD_{typ}"].min()) / (combination_df[f"CMD_{typ}"].max() -
                                                                  combination_df[f"CMD_{typ}"].min()))

        # Plot
        for k, df in enumerate(combination_dfs[:]):

            fig, ax = plt.subplots(1, 1, figsize=(4, 6))

            plt.scatter(OCs[k].density_x, OCs[k].density_y, **OCs[k].kwargs_CMD, label="Pleiades")
            plt.plot(df["m_x"], df["m_y"], color="orange", lw=2.5, label="New")
            plt.plot(sim_CMD.cax, sim_CMD.abs_G, color="magenta", label="old")
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
            ax.set_xlabel(f"{sim_CMD.cols[0]}", labelpad=1, color="white")
            ax.set_ylabel(f"{sim_CMD.cols[1]} (mag)", labelpad=1, color="white")
            ax.set_ylim(15, -2)
            ax.set_xlim(-0.5, 4)
            # title
            plt.title(
                f"Difference = {round(combination_df.loc[k, f'scaled_CMD_{typ}'], 2)} \n {combination_df.iloc[k].values[:4]}",
                color="white", y=1.01)
            plt.subplots_adjust(left=0.18, right=0.98)
            # legend
            legend = ax.legend(loc="best", edgecolor="white", facecolor="black")
            for text in legend.get_texts():
                text.set_color('white')

            if save_plot:
                fig.savefig(output_path + f"{c}_CMD_{typ}_gp_{k}_plx_{combination_df.iloc[k].values[0]}_bin_"
                                          f"{combination_df.iloc[k].values[1]}_ext_{combination_df.iloc[k].values[2]}_"
                                          f"field_{combination_df.iloc[k].values[3]}.png", dpi=300)

            plt.close()

    all_dfs.append(combination_df)

    fig, stats = anova_analysis(cluster_name=c, df=combination_df, color_palette=palette_3, output_path=output_path)

    fig.show()




'''
# Plot the results
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 15))

for i_row, c in enumerate(clusters[:1]):

    for i_col, param in enumerate(["u_plx", "f_binaries", "extinction", "f_field"]):
        # Group by "extinction" and calculate the mean of "CMD1" for each group
        # averaged_1 = all_dfs[i_row].groupby(param)['scaled_CMD_1']#.mean()
        # averaged_2 = all_dfs[i_row].groupby(param)['scaled_CMD_2']#.mean()
        # averaged_3 = all_dfs[i_row].groupby(param)['scaled_CMD_3']#.mean()

        group_data = all_dfs[i_row].groupby(param)['scaled_CMD_1']
        for group in group_data:
            ax[i_row][i_col].boxplot(group, posix=group.index)

        # ax[i_row][i_col].plot(averaged_1.index, averaged_1.values, label="BP-RP")
        # ax[i_row][i_col].plot(averaged_2.index, averaged_2.values, label="BP-G")
        # ax[i_row][i_col].plot(averaged_3.index, averaged_3.values, label="G-RP")
        # ax[i_row][i_col].set_xlabel(param)
        ax[i_row][i_col].set_ylabel('Difference btw curves')
        ax[i_row][i_col].set_ylim(0, 1)
        ax[i_row][1].set_title(c)

    plt.legend(loc="best")

plt.tight_layout()

# plt.savefig(output_path + "All_parameter_variations.png", dpi=300)

plt.show()
'''

print("Routine executed sucessfully.")
