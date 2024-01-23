import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list, cluster_name_list

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# load the Pleiades data from CI
CI_clusters = cluster_name_list[0]
CI_df = cluster_df_list[0]
OC = star_cluster(CI_clusters[26], CI_df)

# create CMD and PCA scatter data
OC.create_CMD()
x_pca_density, y_pca_density, kw = CMD_density_design(OC.PCA_XY, cluster_obj=OC, density_plot=False)

# reduce marker size for paper
OC.kwargs_CMD["s"] = 20
kw["s"] = 20
colors = ["#7fc97f", "#e7298a"]
labels = ["c1", "c2"]
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(3.54399, 3), gridspec_kw={'width_ratios': [1, 1]})
ax = plt.subplot2grid((1, 2), (0, 0))
plt.subplots_adjust(left=0.06, bottom=0.143, top=0.93, wspace=0.12, right=0.98)

# Subplot 1
ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)

# define the multiplication factor of the arrows
comp_BU = [1, 50]
for i, (comp, var) in enumerate(zip(OC.pca.components_, OC.pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    comp_blown = comp * comp_BU[i]  # blow up second comp for visibility

    # annotate the arrows
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
ax.set_ylabel(r"M$_{\rm G}$ (mag)")
ax.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$ (mag)", labelpad=2)
ax.set_title("CMD", fontsize=10)

# Subplot 2
ax1 = plt.subplot2grid((1, 2), (0, 1))
ax1.scatter(x_pca_density, y_pca_density, label="Pleiades", **kw)
ax1.set_ylabel("PCA Y")
ax1.set_xlabel("PCA X")
ax1.set_title("PCA space", fontsize=10)
ax1.get_xlim()
ax1.get_ylim()
ax1_h, labels = ax1.get_legend_handles_labels()

# custom designed legend
legend_elements = [mpatches.Patch(color=colors[0], label='Comp. 1'),
                   mpatches.Patch(color=colors[1], label='Comp. 2'),
                   Line2D([0], [0], marker='.', color=None, lw=0, label='Pleiades',
                          markerfacecolor=kw["cmap"](0.7), markeredgecolor=kw["cmap"](1), markersize=10),
                   Line2D([0], [1], ls='solid', color="black", lw=1, label='PCA axes')
                   ]
ax1.legend(handles=legend_elements, loc="upper right")

plt.show()
if save_plot:
    fig.savefig(output_path + "PCA_components_{}.pdf".format(OC.name), dpi=600)
# ----------------------------------------------------------------------------------------------------------------------