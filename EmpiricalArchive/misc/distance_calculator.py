import matplotlib.pyplot as plt
import seaborn as sns
from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *

from EmpiricalArchive.IsoModulator.comparison_function import interpolate_single_isochrone
from sklearn.neighbors import NearestNeighbors

# 0.1 Set the correct output paths
output_path = my_utility.set_output_path()
results_path = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/"
isochrone_path = "data/Isochrones/Empirical/"

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 12

mastertable_path = "/Users/alena/PycharmProjects/Empirical_Isochrones/EmpiricalArchive/data/Isochrones/"
mastertable = pd.read_csv(
    mastertable_path + "Mastertable_Archive_TWA_Carina.csv")  # extended mastertable incl Carina and TWA

######################
cluster = "Carina"
######################

# data points
if cluster == "TW_Hydrae":
    datapoints = pd.read_csv("../data/Cluster_data/TW_Hydrae/TWA_single_p68_w_errors_DR32016.csv")
elif cluster == "Carina":
    Carina_all_df = pd.read_csv("../data/Cluster_data/Carina/Carina_sources.csv")
    Carina_all_df.rename(columns={"parallax": "Plx", "parallax_error": "e_Plx"}, inplace=True)
    datapoints = Carina_all_df[Carina_all_df["bonafide_UVES"] == True]
else:
    raise ValueError("Name issue")

datapoints["BPRP"] = datapoints["BPmag"] - datapoints["RPmag"]
datapoints["BPG"] = datapoints["BPmag"] - datapoints["Gmag"]
datapoints["GRP"] = datapoints["Gmag"] - datapoints["RPmag"]
datapoints["distance"] = 1000 / datapoints["Plx"]
datapoints["abs_G"] = datapoints["Gmag"] - 5 * np.log10(datapoints["distance"]) + 5

cluster_df = mastertable[mastertable["Cluster"] == cluster]  # filter for the example clusters
cluster_names = np.unique(mastertable["Cluster"])

distance_df = pd.DataFrame(
    columns=["Cluster", "distance_BPRP", "distance_BPG", "distance_GRP", "ref_age", "ref_age_Myr"],
    dtype=float)
distance_df['Cluster'] = distance_df['Cluster'].astype(str)

for n, name in enumerate(cluster_names[:]):
    df = mastertable[mastertable["Cluster"] == name]
    distance_df.loc[n, "Cluster"] = name
    if len(np.unique(df["ref_age"])) < 2:
        distance_df.loc[n, "ref_age"] = np.unique(df["ref_age"])[0]
        distance_df.loc[n, "ref_age_Myr"] = (10 ** np.unique(df["ref_age"])[0]) / 1e6
    else:
        raise ValueError(f"Encountered more than one reference age for cluster: {name}")

    for band_id in ["BPRP", "BPG", "GRP"]:

        bands = [f'{band_id}_isochrone_x', f'{band_id}_isochrone_y']
        band_df = df[bands]
        nona_df = band_df.dropna()
        if band_df.shape[0] != nona_df.shape[0]:
            # print(name, band_id)
            band_df = nona_df
        interp_points_new = interpolate_single_isochrone(band_df[bands[0]], band_df[bands[1]],
                                                         100_000)
        interp_points_original = interpolate_single_isochrone(cluster_df[bands[0]],
                                                              cluster_df[bands[1]], 100)

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

        distance_df.loc[n, f"distance_{band_id}"] = final_result  # store distance in grid dataframe

for band_id in ["BPRP", "BPG", "GRP"]:
    # Apply rescaling to the 'value' column
    distance_df[f"scaled_distance_{band_id}"] = distance_df[f"distance_{band_id}"].apply(
        lambda x: (x - distance_df[f"distance_{band_id}"].min()) / (distance_df[f"distance_{band}"].max() -
                                                                    distance_df[f"distance_{band}"].min()))

min_indices = distance_df[distance_df["Cluster"] != cluster].filter(like='scaled_distance').idxmin()

# Get corresponding rows
min_rows = distance_df.loc[min_indices]

print(min_rows)

bands = ["BPRP", "BPG", "GRP"]
colors = ["#e7298a", "#e7298a", "#e6ab02"]

fig, ax = plt.subplots(1, 3, figsize=(8, 4))

for i in range(3):
    band = bands[i]

    ax[i].scatter(datapoints[band], datapoints["abs_G"], color="black", s=30, marker=".", label=f"stars")
    ax[i].plot(cluster_df[f"{band}_isochrone_x"], cluster_df[f"{band}_isochrone_y"], label=cluster)
    ax[i].plot(mastertable[mastertable["Cluster"] == min_rows.iloc[i].Cluster][f"{band}_isochrone_x"],
               mastertable[mastertable["Cluster"] == min_rows.iloc[i].Cluster][f"{band}_isochrone_y"], label=
               f'{min_rows.iloc[i].Cluster}\n{round(min_rows.iloc[i].ref_age_Myr, 1)} Myr', color=colors[i])
    ax[i].set_xlabel(band)

    ax[i].set_ylim(ax[i].get_ylim()[1], ax[i].get_ylim()[0])
    ax[i].set_title(f"Deviation: {round(min_rows.iloc[i][f'scaled_distance_{band}'], 2)}")

plt.suptitle(cluster)
ax[0].set_ylabel("abs G")

# Initialize empty dictionaries to store unique handles and labels
unique_handles = {}
unique_labels = {}

# Grab handles and labels from subplots and store unique ones
for ax in ax.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in unique_labels:
            unique_handles[label] = handle
            unique_labels[label] = label

# Create a single legend
plt.legend(unique_handles.values(), unique_labels.values(), bbox_to_anchor=(1,1,-0.2,0))

plt.subplots_adjust(left=0.07, bottom=0.12, right=0.79, top=0.88)
fig.show()

fig.savefig(output_path+f"{cluster}_empirical_isochrone_distance.png", dpi = 300)
