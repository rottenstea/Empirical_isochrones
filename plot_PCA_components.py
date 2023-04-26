import my_utility

# Py-scripts
from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# output path
output_path = my_utility.set_output_path()
sns.set_style("darkgrid")
save_plot = False

CI_clusters = cluster_name_list[0]
CI_df = cluster_df_list[0]

OC = star_cluster(CI_clusters[26], CI_df)
OC.create_CMD()
x_pca_density, y_pca_density, kw = CMD_density_design(OC.PCA_XY, cluster_obj=OC, density_plot=False)

colors = ["#7fc97f", "#fdc086"]
labels = ["c1", "c2"]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [1, 1]})
sns.set(font_scale=1.4)

ax = plt.subplot2grid((1, 2), (0, 0))
ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
comp_BU = [1, 15]
for i, (comp, var) in enumerate(zip(OC.pca.components_, OC.pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    comp_blown = comp * comp_BU[i]
    # print(comp_scaled)
    ax.annotate('', OC.pca.mean_, OC.pca.mean_ + comp, arrowprops=dict(arrowstyle="-", color=colors[i], linewidth=4,
                                                                       shrinkA=0, shrinkB=0))
    ax.annotate('', OC.pca.mean_, OC.pca.mean_ + comp_blown,
                arrowprops=dict(arrowstyle="<-", color="black", linewidth=1,
                                shrinkA=0, shrinkB=0))
plt.gca().set(
    aspect="equal",
)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)
ax.set_xlim(-1, 5)
ax.set_ylabel(r"M$_{\rm G}$")
ax.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
ax.set_title("CMD")

ax1 = plt.subplot2grid((1, 2), (0, 1))
ax1.scatter(x_pca_density, y_pca_density, label="Sources", **kw)
ax1.set_ylabel("PCA Y")
ax1.set_xlabel("PCA X")
ax1.set_title("PCA space")
ax1.get_xlim()
ax1.get_ylim()
ax1_h, labels = ax1.get_legend_handles_labels()

legend_elements = [mpatches.Patch(color=colors[0], label='Comp. 1'),
                   mpatches.Patch(color=colors[1], label='Comp. 2'),
                   Line2D([0], [0], marker='.', color=None, lw=0, label='Sources',
                          markerfacecolor=kw["cmap"](0.7), markeredgecolor=kw["cmap"](1), markersize=10),
                   Line2D([0], [1], ls='solid', color="black", lw=1, label='Axis direction')
                   ]
ax1.legend(handles=legend_elements, loc="best")
plt.suptitle(OC.name.replace("_", " "))
plt.subplots_adjust(left=0.03, right=0.95, bottom=0.12, wspace=0.1, hspace=0.1)

plt.show()
if save_plot:
    plt.savefig(output_path + "{}_PCA_components.pdf".format(OC.name), dpi=500)
