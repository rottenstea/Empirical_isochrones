import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import CIII_df, CIII_clusters_new
from EmpiricalArchive.Extraction.Empirical_iso_reader import merged_BPRP


def linear_function(slope, variable, x0):
    """
    Linear function
    """
    return slope * variable + x0


def WD_filter(df_input, cols):
    """
    Cut out the WD component of clusters in the lower left corner of a CMD. Applicable to either one cluster or the
    whole catalog.
    :param df_input: cluster data
    :param cols: CMD columns
    :return: WD-filtered dataframe
    """
    # drop possible NaN values
    df = df_input.dropna()

    # Create a boolean WD list (df column did not work out)
    WD = []

    # turn dataframe column to numpy array
    y_col = df[cols[0]].to_numpy()
    x_col = (df[cols[1]] - df[cols[2]]).to_numpy()

    # Define the linear function variables
    intercept = 0.5 * max(y_col)
    k = (0.5 * max(y_col) - max(y_col)) / (min(x_col) - 0.5 * max(x_col))

    # Iterate over each point in the array
    for i, x in enumerate(x_col):
        y = y_col[i]

        # Calculate the y value of the linear function at the given x coordinate
        line_y = linear_function(k, x, intercept)

        # Compare the y value of the linear function with the y coordinate of the point
        if y < line_y:  # don't forget the inverted Y-axis
            WD.append(0)

        # if condition is fulfilled set the WD flag to True
        elif y >= line_y:
            WD.append(1)
            # print(f"({x}, {y}) is below the line y = {intercept} - {k} *x")
        else:
            print(f"Problem encountered at index {i}, for Y value of {y}.")

    # Set completed list to boolean flag array in the dataframe
    df["WD"] = WD

    # Use the bool WD column to filter the input dataframe
    res = df[df["WD"] == 0]
    return res


# test of WD filter

# import matplotlib.pyplot as plt
# MG1_sin_WD = WD_filter(cluster_df_list[5], cols=["Gmag", "Gmag", "RPmag"])
# g = plt.figure()
# plt.scatter(cluster_df_list[5]["Gmag"] - cluster_df_list[5]["RPmag"], cluster_df_list[5]["Gmag"])
# plt.scatter(MG1_sin_WD["Gmag"] - MG1_sin_WD["RPmag"], MG1_sin_WD["Gmag"], marker="x")
# plt.gca().invert_yaxis()
# g.show()

# Check influence of WD using CIII clusters
# mostly a copy of the main.py file

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
HP_file = "//data/Hyperparameters/White_dwarf_test.csv"
my_utility.setup_HP(HP_file)
kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA + PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# use White dwarf filter function
CIII_no_WD = WD_filter(CIII_df, cols=["Gmag", "BPmag", "RPmag"])

for cluster in CIII_clusters_new:

    # Create a class object for each cluster and the CMD
    OC = star_cluster(cluster, CIII_no_WD)
    OC.create_CMD()

    # Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file, True)
    except IndexError:
        print(OC.name)
    curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 1000
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs,
                                           output_loc="/Users/alena/PycharmProjects/Empirical_Isochrones/"
                                                      "data/Isochrones/White_dwarf_filter_test/")

    # Plot the result
    fig = CMD_density_design(OC.CMD, cluster_obj=OC)

    plt.plot(result_df["l_x"], result_df["l_y"], ls="--", color="grey", label="5. perc")
    plt.plot(result_df["m_x"], result_df["m_y"], ls="--", color="orange", label="Isochrone")
    plt.plot(result_df["u_x"], result_df["u_y"], ls="--", color="grey", label="95. perc")

    # Plot comparison with normal isochrones
    archive_iso = merged_BPRP[merged_BPRP["Cluster_id"] == cluster]
    plt.plot(archive_iso["l_x"], archive_iso["l_y"], ls="solid", color="black", label="5. perc")
    plt.plot(archive_iso["m_x"], archive_iso["m_y"], ls="solid", color="red", label="Isochrone")
    plt.plot(archive_iso["u_x"], archive_iso["u_y"], ls="solid", color="black", label="95. perc")

    plt.show()
    if save_plot:
        fig.savefig(output_path + f"{OC.name}_no_WD.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
