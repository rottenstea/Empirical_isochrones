import os
from datetime import date

from pre_processing import cluster_df_list, cluster_name_list

from Classfile import *
import seaborn as sns
import matplotlib.pyplot as plt

"""
Code structure for the Summary plot in 2D. Everything is running smoothly here. Also the raster plot for all clusters
is built in here.
"""

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

HP_file = "data/Hyperparameters/test_temp.csv"

# Catalog import
# ----------------------------------------------------------------------------------------------------------------------
cluster_df_list.pop(1)
cluster_name_list.pop(1)

Archive_clusters = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(cluster_df_list, axis=0)

sns.set_style("darkgrid")
Fig, vax2 = plt.subplots(figsize=(12, 23), nrows=10, ncols=7)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

iso_array = []
N_stars = []
cluster_w_stars = []
j = 0

ages_archive = []

for i, cluster in enumerate(Archive_clusters[:5]):
    OC = star_cluster(cluster, Archive_df)

    if OC.Nstars > 96:
        cluster_w_stars.append(OC)
        if not np.isnan(OC.data["ref_age"].unique()[0]):
            ages_archive.append(OC.data["ref_age"].unique()[0])
        else:
            if not np.isnan(OC.data["age_D"].unique()[0]):
                ages_archive.append(OC.data["age_D"].unique()[0])
            else:
                print(OC.name)
                ages_archive.append(0)

        N_stars.append(OC.Nstars)

        OC.create_CMD()

        curve, iso = OC.curve_extraction(svr_data=OC.PCA_XY, HP_file=HP_file)
        iso_array.append(iso)

for i, ax in enumerate(vax2.flat[:]):
    try:

        OC = cluster_w_stars[i]
        sns.set_style("darkgrid")

        cs = ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
        ax.plot(iso_array[i][:, 0], iso_array[i][:, 1], color="red")

        ax.set_title(OC.name.replace("_", " "))
        if i in range(0, 70, 7):
            ax.set_ylabel(f"absolute {OC.CMD_specs['axes'][0]}")

        if i in range(63, 70):
            ax.set_xlabel(OC.CMD_specs["axes"][1])

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)

        plt.colorbar(cs, ax=ax)

    except IndexError:
        pass

    # plt.subplots_adjust(top=0.92, left=0.1, right=0.95, bottom=0.12)
# for ax in range(67, 72):
#    Fig.delaxes(vax2.flat[ax])

plt.subplots_adjust(wspace=0.5, hspace=0.4)

Fig.show()
# Fig.savefig(output_path + "composite_Archive_OCPCAXY_normweights.pdf", dpi=500)


Fig2 = plt.figure(figsize=(7, 5))

ax1 = plt.subplot2grid((1, 1), (0, 0))
# ax2 = plt.subplot2grid((1, 2), (0, 1))
palette = sns.color_palette("crest", len(ages_archive))  #
palette_cmap = sns.color_palette("crest_r", as_cmap=True)  #

idx_es = np.arange(len(ages_archive))

id_ages = np.stack([idx_es, ages_archive], axis=1)

sorted_ages = id_ages[(-id_ages[:, 1]).argsort()]

# print(sorted_ages)


for k, element in enumerate(sorted_ages[:-1, 1]):
    # for clus in ScoCen_clusters:
    # print(element)
    cluster_id = int(sorted_ages[k, 0])
    CMD_isochrone = iso_array[cluster_id]
    # kr = int(0.05 * len(CMD_isochrone[:, 0]))
    kr = 0
    ax1.plot(CMD_isochrone[kr:, 0], CMD_isochrone[kr:, 1], color=palette[k], lw=0.5, label=Archive_clusters[k])

ymin, ymax = ax1.get_ylim()
ax1.set_ylim(16, -3)
ax1.set_xlim(-0.5, 5)
ax1.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
ax1.set_ylabel(r"M$_{\rm G}$")
# Fig.subplots_adjust(right=0.8, wspace=0.2)
# cbar_ax = Fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap=palette_cmap,
                           norm=plt.Normalize(vmin=np.min(sorted_ages[:-1, 1]), vmax=np.max(sorted_ages[:-1, 1])))
plt.colorbar(sm)
Fig2.show()
# Fig2.savefig(output_path+"Age_summary_Archive_OCPCAXY.pdf", dpi = 500)
