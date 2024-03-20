from scipy.integrate import simps
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


def resolve_duplicates(x: np.array):
    """
    Resolves possible duplicates in color and absolute magnitude of the isochrone data arrays that would hinder the interpolation.

    :param x:
    :return:
    """
    unique, counts = np.unique(x, return_counts=True)
    dup = unique[counts > 1]
    # Add small random effect
    x += np.isin(x, dup).astype(np.float64) * np.random.normal(0, 1e-5, x.size)
    return x


def interpolate_single_isochrone(color, abs_mag, nb_interpolated):
    # Interpolated points along line:
    alpha = np.linspace(0, 1, nb_interpolated)
    has_potential_duplicates = True
    while has_potential_duplicates:
        points = np.vstack([color, abs_mag]).T
        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        try:
            interpolator = interp1d(distance, points, kind='slinear', axis=0)
            has_potential_duplicates = False
        except ValueError:
            # If x or y has duplicates we perturb the data slightly
            color = resolve_duplicates(color)
            abs_mag = resolve_duplicates(abs_mag)
    # Interpolated points is 2d output
    interpolated_points = interpolator(alpha)
    # Interpolate mass along line
    return interpolated_points


def compute_NN_distance(df_new, old_x, old_y):
    interp_points_new = interpolate_single_isochrone(df_new["m_x"], df_new["m_y"], 100000)
    interp_points_original = interpolate_single_isochrone(old_x, old_y, 100)
    # fig_test = plt.figure()
    # plt.scatter(interp_points_new[:, 0], interp_points_new[:, 1], label="new")
    # plt.scatter(interp_points_original[:, 0], interp_points_original[:, 1], label="original")
    # plt.legend(loc="best")
    # fig_test.show()

    # Number of neighbors to consider
    k_neighbors = 100

    # Fit nearest neighbor model on new array
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(interp_points_new)

    # Find distances and indices of the nearest neighbors of each point in original isochrone
    distances, indices = nbrs.kneighbors(interp_points_original)

    # Calculate average distance for each point in the original array
    average_distances = np.mean(distances, axis=1)

    # Sum over all the averages for the 100 points of the original array
    final_result = np.sum(average_distances)

    return final_result


# ------ deprecated -----------
def compute_areas(df_new, x_old, y_old):
    # Interpolate the second curve onto the x values of the first curve
    y2_interp = np.interp(df_new['m_x'], x_old, y_old)

    # Calculate the absolute difference between the two curves
    # difference = np.abs(result_df["m_y"] - y2_interp)
    euclidean_distances = np.sqrt(df_new["m_y"] - y2_interp) ** 2

    # Calculate the area between the curves using the trapezoidal rule
    area_between_curves = simps(euclidean_distances, df_new["m_x"])

    print("Area between curves:", area_between_curves)

    return area_between_curves


# function to plot the results as box-line and return df of the medians
def anova_analysis(cluster_name, df, color_palette="Blues", output_path=None):
    stats_df = pd.DataFrame(columns=["Cluster", "variable", "CMD_type", "value"])
    df = df.drop(0)
    # Define the columns to keep as identifiers
    id_vars = ['u_plx', 'f_binaries', 'extinction', 'f_field']

    # Melt the DataFrame to combine CMD_1, CMD_2, CMD_3 into a single column "CMD"
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=['scaled_CMD_1', 'scaled_CMD_2', 'scaled_CMD_3'],
                        var_name="CMD_type", value_name="Diff")

    # Extract CMD_type number from 'CMD_type'
    melted_df['CMD_type'] = melted_df['CMD_type'].str.extract(r'(\d+)').astype(int)
    # Define the mapping dictionary
    cmd_mapping = {1: 'BP-RP', 2: 'BP-G', 3: 'G-RP'}
    # Map the numeric values to their corresponding string representations
    melted_df['CMD_type'] = melted_df['CMD_type'].map(cmd_mapping)

    fig, ax = plt.subplots(2, 4, figsize=(7.58, 5.5), sharey=True)

    axes = ax.ravel()

    df_id = 0
    for i, var in enumerate(id_vars):

        # Group the DataFrame by the f_field column
        grouped_df = melted_df.groupby(var)

        # Combine the data for all groups
        combined_data = pd.concat([grouped_df.get_group(val) for val in grouped_df.groups])

        if i != 3:
            # boxplot
            sns.boxplot(data=combined_data, x=var, y="Diff", hue="CMD_type", palette=color_palette, legend=False,
                        ax=axes[i]).set(title=f'{var}', xlabel="", ylabel='Normalized difference', ylim=[-0.1, 1.1])
            # lineplot
            sns.lineplot(data=combined_data, x=var, y="Diff", hue="CMD_type", estimator=np.median, palette=color_palette,
                         legend=False, ax=axes[i + 4]).set(xlabel="", ylabel='Normalized difference', ylim=[-0.1, 1.1])
        else:
            # boxplot
            sns.boxplot(data=combined_data, x=var, y="Diff", hue="CMD_type", palette=color_palette, legend=True,
                        ax=axes[i]).set(title=f'{var}', xlabel="", ylabel='Normalized difference', ylim=[-0.1, 1.1])
            # lineplot
            sns.lineplot(data=combined_data, x=var, y="Diff", hue="CMD_type", estimator=np.median, palette=color_palette,
                         legend=False, ax=axes[i + 4]).set(xlabel="", ylabel='Normalized difference', ylim=[-0.1, 1.1])
            # axes[i+4].legend(bbox_to_anchor=(1, 1))

        for typ in np.unique(combined_data["CMD_type"]):
            df_G = combined_data[combined_data["CMD_type"] == typ]["Diff"]
            stats_df.loc[df_id, "Cluster"] = cluster_name
            stats_df.loc[df_id, "variable"] = var
            stats_df.loc[df_id, "CMD_type"] = typ
            stats_df.loc[df_id, "value"] = np.median(df_G)

            df_id += 1

    axes[1].set_xlabel("gridpoints", x=1.1)
    axes[5].set_xlabel("gridpoints", x=1.1)

    axes[3].legend(bbox_to_anchor=(1, -0.9, 1, 1))
    # axes[7].legend(bbox_to_anchor=(1, 1))

    plt.subplots_adjust(wspace=0.15, hspace=0.3, top=0.9)
    plt.suptitle(cluster_name.replace("_", " "))

    if output_path is not None:
        fig.savefig(output_path + f'{cluster_name}_box-line_Grid.png', dpi=300)
        stats_df.to_csv(output_path+f'{cluster_name}_medians.csv')

    return fig, stats_df