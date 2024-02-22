import matplotlib.pyplot as plt
import seaborn as sns
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import CII_df, CII_clusters

from EmpiricalArchive.My_tools import my_utility

"""
Just a short script checking the differences between weighted and unweighted SVR.
"""
output_path = my_utility.set_output_path()

HP_file = "//data/Hyperparameters/Weight_test.csv"
my_utility.setup_HP(HP_file)

sns.set_style("darkgrid")
save_plot = False

for i, cluster in enumerate(CII_clusters):
    OC = star_cluster(cluster, CII_df)

    OC.create_CMD()

    # 1. Weighted case (default)
    curve_w, isochrone_w = OC.curve_extraction(svr_data=OC.PCA_XY, HP_file=HP_file, always_tune=True)

    # 2. Unweighted case
    weights = np.ones(len(OC.PCA_XY[:, 0]))
    curve_u, isochrone_u = OC.curve_extraction(svr_data=OC.PCA_XY, HP_file=HP_file, svr_weights=weights,
                                               always_tune=True)

    fig1 = plt.figure(figsize=(4, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))

    cm = plt.cm.get_cmap("crest")
    sc = ax.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, s=20)

    ax.plot(isochrone_w[:, 0], isochrone_w[:, 1], color="red", label="weighted")
    ax.plot(isochrone_u[:, 0], isochrone_u[:, 1], color="orange", label="unweighted")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel(OC.CMD_specs["axes"][0])
    ax.set_xlabel(OC.CMD_specs["axes"][1])
    ax.set_title(OC.name)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper right")
    plt.colorbar(sc)
    fig1.show()
    if save_plot:
        fig1.savefig(output_path + "Weight_comparison_{0}_{1}.pdf".format(OC.name, OC.CMD_specs["short"]), dpi=500)
