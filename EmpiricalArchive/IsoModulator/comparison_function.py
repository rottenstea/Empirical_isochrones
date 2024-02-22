from scipy.integrate import simps
from scipy.interpolate import interp1d
import numpy as np

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

    interp_points_new = interpolate_single_isochrone(df_new["m_x"], df_new["m_y"], 10000)
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
