import os
from datetime import date

from pre_processing import *
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
# ----------------------------------------------------------------------------------------------------------------------
Pleiades = "data/Cluster_data_raw/Pleiades_w_plx.csv"

df_columns = ["Cluster", "plx", "umag", "gmag", "rmag", "imag", "Ymag", "Jmag", "Hmag", "Kmag"]

new_colnames = ["Cluster_id", "plx", "umag", "gmag", "rmag", "imag", "Ymag", "Jmag", "Hmag", "Kmag"]

Pleiades_cluster, Pleiades_df = create_df(Pleiades, columns=df_columns, names=new_colnames)

Pleiades_filtered_df = Pleiades_df[Pleiades_df["imag"] > 13]
# ----------------------------------------------------------------------------------------------------------------------
J, H, K = np.genfromtxt("data/PARSEC_isochrones/Nuria_clusters/2MASS_30Myr.txt", usecols=(-3, -2, -1), unpack=True)
i = np.genfromtxt("data/PARSEC_isochrones/Nuria_clusters/panSTARRs1_30Myr.txt", usecols=(-4))

IC4665 = "data/Cluster_data_raw/IC_4665_w_plx.csv"

df_columns = ["Cluster", "median_plx", "g", "r", "i", "z", "y", "J", "H", "K"]

new_colnames = ["Cluster_id", "plx", "gmag", "rmag", "imag", "zmag", "Ymag", "Jmag", "Hmag", "Kmag"]

IC4665_cluster, IC4665_df = create_df(IC4665, columns=df_columns, names=new_colnames)

IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13) & (IC4665_df["imag"] < 99) & (IC4665_df["Kmag"] < 99)]

N_clusters = np.concatenate([IC4665_cluster, Pleiades_cluster])
N_df = pd.concat([IC4665_filtered_df, Pleiades_filtered_df], axis=0)
# ----------------------------------------------------------------------------------------------------------------------
sns.set_style("darkgrid")
colors = ["red", "darkorange"]

params1 = {'C': 100.0, 'epsilon': 0.01353876180022544, 'gamma': 'scale', 'kernel': 'rbf'}
params2 = {'C': 37.92690190732246, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'}
params = [params1, params2]

fig1 = plt.figure(figsize=(4, 6))
ax = plt.subplot2grid((1, 1), (0, 0))

for i, cluster in enumerate(N_clusters[:]):
    OC = star_cluster(cluster, N_df, CMD_parameters=["imag", "imag", "Kmag"])
    h = CMD_density_design([OC.CMD[:, 0], OC.CMD[:, 1]], title=OC.name)
    h.show()

    OC.kwargs_CMD["s"] = 50

    pca = PCA(n_components=2)
    pca_arr = pca.fit_transform(OC.CMD)

    evals = np.logspace(-2, -1.5, 20)
    # gvals = np.logspace(-4, -1, 50)
    Cvals = np.logspace(-2, 2, 20)

    param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals,
                      epsilon=evals)

    # params = SVR_Hyperparameter_tuning(pca_arr, param_grid)
    # params = {'C': 37.92690190732246, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'}   #i-k band
    # {'C': 0.29763514416313175, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'}        # J-K band

    svr = SVR(**params[i])

    svr_predict = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)

    X = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)
    Y = pca_arr[:, 1]

    Y_all = svr.fit(X, Y).predict(svr_predict)
    print("SVR Test score:", svr.score(svr_predict, Y.ravel()))

    SVR_all = np.stack([svr_predict[:, 0], Y_all], 1)
    SVR_all = SVR_all[SVR_all[:, 0].argsort()]
    rev_transform = pca.inverse_transform(SVR_all)
    kr = 0

    if i == 0:
        ax.scatter(OC.density_x, OC.density_y, label="IC_4665 data", **OC.kwargs_CMD)
    elif i == 1:
        OC_density_x, OC_density_y, OC_kwargs = CMD_density_design([OC.CMD[:, 0], OC.CMD[:, 1]],
                                                                   to_RBG=[0.27, 0.27, 0.27],
                                                                   from_RBG=[0.74, 0.74, 0.74], density_plot=False)
        ax.scatter(OC_density_x, OC_density_y, label="Pleiades data", **OC_kwargs)
    ax.plot(rev_transform[kr:, 0], rev_transform[kr:, 1], color=colors[i], label="{}".format(OC.name))
    # ax2.set_ylabel(r"M$_{\rm G}$")

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)

ax.set_ylabel(r"abs mag i")

ax.set_xlabel(r"${\rm i}$ - ${\rm K}$")
# ax2.legend(loc="best", fontsize=16)
ax.set_title("Empirical isochrones")  # , y=0.97)
# a = 100
# plt.plot(i[:a] - K[:a], i[:a], color="orange", label = "PARSEC")

plt.legend(bbox_to_anchor=(1, 1), loc="upper right")

fig1.show()
fig1.savefig(output_path + "Comparison_Pleiades_IC4665_data"
                           ".png", dpi=500)

