from Classfile import *

from pre_processing import create_df
from version_I.Support_Vector_Regression import *
from sklearn.decomposition import PCA

CII_raw = "data/Cluster_data/all_ages/CatalogII_BCD_ages.csv"

CII_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
            "RUWE"]

CII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

CII_clusters, CII_df = create_df(CII_raw, CII_cols, CII_names, q_filter)
OC = star_cluster(CII_clusters[0], CII_df)
OC.create_CMD()
weights = OC.create_weights()

pca = PCA(n_components=2)
pca_arr = pca.fit_transform(OC.CMD)

evals = np.logspace(-2, -1.5, 20)
Cvals = np.logspace(-2, 2, 20)

param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals,
                  epsilon=evals)

params = {'C': 14.38449888287663, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'}

svr = SVR(**params)

svr_predict = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)

X = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)
Y = pca_arr[:, 1]

Y_all1 = svr.fit(X, Y).predict(svr_predict)

print("SVR Test score:", svr.score(svr_predict, Y.ravel()))

SVR_all1 = np.stack([svr_predict[:, 0], Y_all1], 1)
SVR_all1 = SVR_all1[SVR_all1[:, 0].argsort()]
rev_transform1 = pca.inverse_transform(SVR_all1)


params2 = SVR_Hyperparameter_tuning(pca_arr, param_grid, weight_data = weights)
print("p2",params2)

svr2 = SVR(**params2)

Y_all2 = svr2.fit(X, Y, sample_weight=weights).predict(svr_predict)
SVR_all2 = np.stack([svr_predict[:, 0], Y_all2], 1)
SVR_all2 = SVR_all2[SVR_all2[:, 0].argsort()]
rev_transform2 = pca.inverse_transform(SVR_all2)


kr = 0

fig1 = plt.figure(figsize=(4, 6))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.scatter(OC.density_x, OC.density_y, label=OC.name, **OC.kwargs_CMD)

ax.plot(rev_transform1[kr:, 0], rev_transform1[kr:, 1], color="red", label="{}".format(OC.name))
ax.plot(rev_transform2[kr:, 0], rev_transform2[kr:, 1], color="orange",ls = "-", label="{}".format(OC.name))
    # ax2.set_ylabel(r"M$_{\rm G}$")

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)

ax.set_ylabel(OC.CMD_specs["axes"][0])
ax.set_xlabel(OC.CMD_specs["axes"][1])


ax.set_title("Empirical isochrone")  # , y=0.97)
# a = 100
# plt.plot(i[:a] - K[:a], i[:a], color="orange", label = "PARSEC")

plt.legend(bbox_to_anchor=(1, 1), loc="upper right")

fig1.show()







