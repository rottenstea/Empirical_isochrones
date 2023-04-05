import matplotlib.pyplot as plt
import seaborn as sns
from Classfile import *
from pre_processing import CII_df, CII_clusters, case_study_dfs, case_study_clusters
from datetime import date
import os

"""
Just a short script checking the differences between weighted and unweighted SVR.
"""

# 0.1 Set the correct output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

HP_file = "data/Hyperparameters/Weight_test.csv"
sns.set_style("darkgrid")

try:
    pd.read_csv(HP_file)
except FileNotFoundError:
    with open(HP_file, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

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
    fig1.savefig(output_path + "Weightcomparison_{0}_{1}.pdf".format(OC.name, OC.CMD_specs["short"]), dpi=500)
