import seaborn as sns
import matplotlib.pyplot as plt

from My_tools import my_utility
from Extraction.Classfile import *
from Extraction.pre_processing import CII_df, CII_clusters

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
# Define the cluster objects for Catalog II
for n, cluster in enumerate(CII_clusters[:]):

    OC = star_cluster(cluster, CII_df)
    errors = OC.create_CMD(return_errors=True)
    e_cax, e_absm = errors[3], errors[4]

    # define number of bins
    bins = 12
    x_err_binned = np.zeros(shape=bins)
    y_err_binned = np.zeros(shape=bins)

    # define even spacing in X and Y directions
    x_spacing = np.linspace(np.min(OC.CMD[:, 0]), np.max(OC.CMD[:, 0]), bins)
    y_spacing = np.linspace(np.min(OC.CMD[:, 1]), np.max(OC.CMD[:, 1]), bins)

    # fill the different bins with the corresponding errors
    i = 0
    for b in range((bins - 1)):
        bin_x, bin_y = [], []
        for idx, star in enumerate(OC.CMD[:, 1]):
            if (star >= y_spacing[i]) and (star < y_spacing[i + 1]):
                bin_x.append(e_cax[idx])
                bin_y.append(e_absm[idx])

        # compute the mean for each bin
        x_err_binned[b] = np.mean(bin_x)
        y_err_binned[b] = np.mean(bin_y)

        i += 1
    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(3.54399, 3))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    axes = [ax1, ax2]

    titlenames = ["binned errors", "5x magnification"]
    ax1.errorbar(x_spacing - 2, y_spacing + 0.5, xerr=x_err_binned, yerr=y_err_binned, color="red", fmt=".",
                 markersize=0,
                 lw=0.5, label="binned errors")
    ax2.errorbar(x_spacing - 2, y_spacing + 0.5, xerr=x_err_binned * 5, yerr=y_err_binned * 5, color="red", fmt=".",
                 lw=0.5, markersize=0, label="5x magnification")

    for q, title in enumerate(titlenames):
        axes[q].scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
        axes[q].set_title(title, fontsize=10)

    ymin, ymax = axes[0].get_ylim()
    xmin, xmax = axes[0].get_xlim()
    axes[0].set_ylim(ymax, ymin)
    axes[0].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymax, ymin)
    axes[1].set_xlim(xmin, xmax)
    axes[0].set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$ (mag)", x=1.)
    axes[0].set_ylabel(r"M$_{\rm G}$ (mag)")
    axes[0].text(-2, 12.5, s=cluster.replace("_", " "))

    ax2.axes.yaxis.set_ticklabels([])
    plt.subplots_adjust(bottom=0.14, left=0.127, right=0.99, wspace=0.05, top=0.93)

    fig.show()
    if save_plot:
        fig.savefig(output_path + "Errorplot_{}.pdf".format(OC.name), dpi=600)
# ------------------------------------------------------------------------------------------------------------------
