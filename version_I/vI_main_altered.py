import os
from datetime import date
from pre_processing import *
from version_I.Confidence_intervals import *
from version_I.old_classfile import *
import seaborn as sns
import matplotlib.pyplot as plt
from version_I.Support_Vector_Regression import *

from sklearn.decomposition import PCA

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

# Hyperparameter path
hypers = os.path.join(output_path, "hyperparams")
try:
    os.mkdir(hypers)
except FileExistsError:
    pass
hypers = hypers + "/"

preprocess = False

# Catalog selection
CI_raw = "/Users/alena/PycharmProjects/PaperI/data/Cluster_data/all_ages/CatalogI_BCD_ages.csv"

CI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
           "G-RP",
           "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
           "RUWE", "Proba"]

CI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability"]

q_filter = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"], "value": [1.4, 0, 0.49]}

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter)

for cluster in CI_clusters[:10]:
    OC = star_cluster(cluster, CI_df, CMD_parameters=["Gmag", "BP-RP"])

    n_boot = 100
    if not preprocess:
        try:
            params = SVR_read_from_file(hypers + "HP_{}.txt".format(OC.name))
        except FileNotFoundError:
            tuned_params = OC.SVR_Hyperparameter_tuning(file_path=hypers + "HP_{}.txt".format(OC.name))
            params = SVR_read_from_file(hypers + "HP_{}.txt".format(OC.name))

        method_kwargs = dict(keys=params, sample_name=OC.name, svr_predict=OC.CMD)
        CMD_isochrone = SVR_PCA_calculation(OC.CMD, **method_kwargs)
        isochrone_store = Isochrone_collection(CMD_isochrone, OC.CMD, n_boot, method_kwargs)
        stats_array = Confidence_interval_stats(CMD_isochrone, isochrone_store)

        x_array = CMD_isochrone[:, 0]
        lower, upper, median = stats_array[:, 0], stats_array[:, 2], stats_array[:, 1]

    else:
        try:
            params = SVR_read_from_file(hypers + "HP_{}.txt".format(OC.name))
        except FileNotFoundError:
            pca = PCA(n_components=2)
            pca_arr = pca.fit_transform(OC.CMD)
            tuned_params = OC.SVR_Hyperparameter_tuning(file_path=hypers + "HP_{}.txt".format(OC.name), PCA=True,
                                                        pca_array=pca_arr)
            params = SVR_read_from_file(hypers + "HP_{}.txt".format(OC.name))

        pca = PCA(n_components=2)
        pca_arr = pca.fit_transform(OC.CMD)
        method_kwargs = dict(keys=params, sample_name=OC.name, pca_case=True, pca_func=pca, svr_predict=pca_arr)
        PCA_isochrone, CMD_isochrone = SVR_PCA_calculation(pca_arr, **method_kwargs)
        isochrone_store = Isochrone_collection(PCA_isochrone, pca_arr, n_boot, method_kwargs, True)

        lower, upper, median = Confidence_interval_stats(PCA_isochrone, isochrone_store, True, pca)
        lower, upper, median = array_sorting([lower, upper, median], 1)
        # np.savetxt(output_path + "{0}_lower_{1}.csv".format(OC.name, n_boot), lower, delimiter=',')
        # np.savetxt(output_path + "{0}_median_{1}.csv".format(OC.name, n_boot), median, delimiter=',')
        # np.savetxt(output_path + "{0}_upper_{1}.csv".format(OC.name, n_boot), upper, delimiter=',')
        CMD_isochrone = CMD_isochrone[CMD_isochrone[:, 1].argsort()]

        # 6. Plot the result for each cluster
    Fig = plt.figure(figsize=(5, 8))
    sns.set_style("darkgrid")
    sns.set(font_scale=1.5)
    ax = plt.subplot2grid((1, 1), (0, 0))

    OC.kwargs_CMD["s"] = 50
    ax.scatter(OC.density_x, OC.density_y, label="Sources", **OC.kwargs_CMD)
    ax.set_ylabel(r"M$_{\rm G}$")

    if not preprocess:
        ax.plot(x_array, lower, label=r"$5^{\rm th}$-p.", color="grey")
        ax.plot(x_array, upper, label=r"$95^{\rm th}$-p.", color="grey")
        ax.fill_between(x_array, lower, upper, color="grey", alpha=0.5)
        ax.plot(x_array, median, label="Isochrone", color="red")

    else:
        # ax.plot(lower[:, 0], lower[:, 1], label=r"$5^{\rm th}$-p.", color="grey")
        #  ax.plot(upper[:, 0], upper[:, 1], label=r"$95^{\rm th}$-p.", color="grey")
        ax.plot(median[:, 0], median[:, 1], label="Isochrone", color="orange")
        ax.plot(CMD_isochrone[:, 0], CMD_isochrone[:, 1], color="red", alpha=0.5)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
    ax.legend(loc="best", fontsize=16)

    plt.subplots_adjust(top=0.92, left=0.16, right=0.95, bottom=0.12)
    plt.suptitle(OC.name.replace("_", " "), y=0.97)
    Fig.show()
        # Fig.savefig(output_path + "{}.png".format(OC.name), dpi=400)
