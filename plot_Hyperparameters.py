# Py-scripts
from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list

# SVR
from sklearn.svm import SVR

# Plotting
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import my_utility

output_path = my_utility.set_output_path()
save_plot = True

HP_file = "data/Hyperparameters/Archive_real.csv"
my_utility.setup_HP(HP_file)

CI, CI_df = cluster_name_list[0], cluster_df_list[0]

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

# Boxplot of variance
# ----------------------------------------------------------------------------
scale = np.empty(shape=len(CI))
for i, cluster in enumerate(CI):
    OC = star_cluster(cluster, CI_df)
    OC.create_CMD()

    X = OC.PCA_XY[:, 0].reshape(len(OC.PCA_XY[:, 0]), 1)
    scale[i] = 1 / (2 * np.var(X))

# f, ax = plt.subplots()
# ax.boxplot(scale)
# plt.show()
# print("median:", np.median(scale), "mean:", np.mean(scale))

# PCA variation and parameter matrix for a sample cluster
# ----------------------------------------------------------------------------

OC = star_cluster(CI[27], CI_df)
OC.create_CMD()

X = OC.PCA_XY[:, 0].reshape(len(OC.PCA_XY[:, 0]), 1)
y = OC.PCA_XY[:, 1]

evals = np.logspace(-4, 0, 5)
gvals = np.logspace(-4, 0, 5)
Cvals = np.logspace(-2, 2, 5)

svrs = []

for val in Cvals:
    svrs.append(SVR(kernel="rbf", C=val, gamma="scale", epsilon=0.1))
for val in gvals:
    svrs.append(SVR(kernel="rbf", C=1, gamma=val, epsilon=0.1))
for val in evals:
    svrs.append(SVR(kernel="rbf", C=1, gamma="scale", epsilon=val))

lw = 2

model_color = ['#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824',
               '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000',
               '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b']

# PCA parameter matrix
# ---------------------

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(7.24551, 5.5), sharey="row", sharex="col",
                         gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.07]})
axes = axes.ravel()

# model line
for ix, svr in enumerate(svrs):

    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw, label="model")
    otherX = X[np.setdiff1d(np.arange(len(X)), svr.support_)]
    otherY = y[np.setdiff1d(np.arange(len(X)), svr.support_)]

    if len(otherX) > 1:
        PC_density_x, PC_density_y, PC_kwargs = \
            CMD_density_design([otherX.reshape(len(otherX), ), otherY.reshape(len(otherY), )],
                               to_RBG=[0.27, 0.27, 0.27], from_RBG=[0.74, 0.74, 0.74], density_plot=False)
        PC_kwargs["s"] = 20

    # support vector scatter
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor=model_color[ix], edgecolor="none", alpha=0.7,
                     marker=".", s=20, label="SV ({})".format(len(X[svr.support_])))
    # other datapoints scatter
    if len(otherX) > 1:
        axes[ix].scatter(PC_density_x, PC_density_y,
                         **PC_kwargs)  # label=" ({})".format(int(len(X) - len(X[svr.support_])))
    # legend
    axes[ix].legend(loc="upper left")

# Colorbars
cax_id = [4, 9, 14]
labels = ["C", "gamma", "epsilon"]

elog = np.arange(-4, 1, dtype=int)
glog = np.arange(-4, 1, dtype=int)
Clog = np.arange(-2, 3, dtype=int)

e_ticks = [r"$10^{%i}$" % i for i in elog]
g_ticks = [r"$10^{%i}$" % i for i in glog]
C_ticks = [r"$10^{%i}$" % i for i in Clog]

hypers = [Cvals, gvals, evals]
hyper_ticks = [C_ticks, g_ticks, e_ticks]

c_ix = 0

for i, hyper in enumerate(hypers):
    divider = make_axes_locatable(axes[cax_id[i]])
    cax_fig = divider.append_axes('right', size=0.05, pad=0.075)
    cmap = mpl.colors.ListedColormap(model_color[c_ix:c_ix + 5])
    norm = mpl.colors.BoundaryNorm(hyper, cmap.N)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                      cax=cax_fig,
                      ticks=hyper,
                      spacing='uniform',
                      orientation='vertical',
                      label=labels[i])
    cb.ax.set_yticklabels(hyper_ticks[i])
    c_ix += 5

axes[2].set_title("C variation")
axes[7].set_title("gamma variation")
axes[12].set_title("epsilon variation")
axes[12].set_xlabel("PCA X")
axes[5].set_ylabel("PCA Y", labelpad=1)
# fig.suptitle("Hyperparameter influence \n (default: C = 1, gamma = 'scale', epsilon = 0.1)", y=0.95, fontsize=16)

plt.subplots_adjust(left=0.075, right=0.92, top=0.96, bottom=0.075, wspace=0.02, hspace=0.18)
# plt.show()
if save_plot:
    fig.savefig(output_path + "HPmatrix_PCA_{}.pdf".format(OC.name), dpi=600)

# CMD parameter matrix
# ---------------------

OC_density_x, OC_density_y, OC_kwargs = CMD_density_design([OC.CMD[:, 0], OC.CMD[:, 1]], to_RBG=[0.27, 0.27, 0.27],
                                                           from_RBG=[0.74, 0.74, 0.74], density_plot=False)
OC_kwargs["s"] = 20

fig_CMD, axes = plt.subplots(nrows=3, ncols=5, figsize=(7.24551, 6.7), sharey="row", sharex="col",
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.07]})
axes = axes.ravel()

for ix, svr in enumerate(svrs):
    SVR_all = np.stack([X[:, 0], svr.fit(X, y).predict(X)], 1)
    SVR_all = SVR_all[SVR_all[:, 0].argsort()]
    rev_transform = OC.pca.inverse_transform(SVR_all)

    # model line
    axes[ix].plot(rev_transform[:, 0], rev_transform[:, 1], color=model_color[ix], lw=lw, label="model")

    # other datapoints
    axes[ix].scatter(OC_density_x, OC_density_y, label="data", **OC_kwargs)
    axes[ix].set_ylim(14, -1)
    # legend
    axes[ix].legend(loc="lower left")

# Colorbars
elog = np.arange(-4, 1, dtype=int)
glog = np.arange(-4, 1, dtype=int)
Clog = np.arange(-2, 3, dtype=int)

e_ticks = [r"$10^{%i}$" % i for i in elog]
g_ticks = [r"$10^{%i}$" % i for i in glog]
C_ticks = [r"$10^{%i}$" % i for i in Clog]

hypers = [Cvals, gvals, evals]
hyper_ticks = [C_ticks, g_ticks, e_ticks]

c_ix = 0

for i, hyper in enumerate(hypers):
    divider = make_axes_locatable(axes[cax_id[i]])
    cax_fig = divider.append_axes('right', size=0.05, pad=0.075)
    cmap = mpl.colors.ListedColormap(model_color[c_ix:c_ix + 5])
    norm = mpl.colors.BoundaryNorm(hyper, cmap.N)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                      cax=cax_fig,
                      ticks=hyper,
                      spacing='uniform',
                      orientation='vertical',
                      label=labels[i])
    cb.ax.set_yticklabels(hyper_ticks[i])
    c_ix += 5

axes[2].set_title("C variation")
axes[7].set_title("gamma variation")
axes[12].set_title("epsilon variation")

axes[5].set_ylabel("M$_{\mathrm{G}}$", labelpad=1)
axes[12].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}_{\mathrm{RP}}$")
# fig_CMD.suptitle("Hyperparameter influence \n (default: C = 1, gamma = 'scale', epsilon = 0.1)", y=0.95, fontsize=16)
plt.subplots_adjust(left=0.071, right=0.922, top=0.966, bottom=0.065, wspace=0.02, hspace=0.15)

fig_CMD.show()
if save_plot:
    fig_CMD.savefig(output_path + "HPmatrix_CMD_{}.pdf".format(OC.name), dpi=600)

# Using different gridsearch methods
# ----------------------------------------------------------------------------
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

params = OC.SVR_read_from_file(file=HP_file)
param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals, epsilon=evals)

# fine_grid = dict(kernel=['rbf'], C=np.linspace(10, 100, 15), gamma=["scale"],
#                epsilon=np.linspace(0.1 * params["epsilon"], 5 * params["epsilon"], 15))
base_estimator = SVR()

sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=3).fit(X, y)
fine_params = sh.best_params_
print(fine_params)
print(sh.best_score_)

from skopt import BayesSearchCV

opt = BayesSearchCV(
    base_estimator, param_grid,
    cv=5)

opt.fit(X, y)

print("val. score: %s" % opt.best_score_)
print("best params: %s" % str(opt.best_params_))
bayes_params = opt.best_params_

svr_tuned = SVR(**params)
svr_untuned = SVR()
svr_finetuned = SVR(**fine_params)
svr_bayes = SVR(**bayes_params)

svrs_new = [svr_untuned, svr_tuned, svr_finetuned, svr_bayes]
colors = ["#7fc97f", "#fdc086", "#beaed4", "#ffff99"]

X = OC.PCA_XY[:, 0].reshape(len(OC.PCA_XY[:, 0]), 1)
y = OC.PCA_XY[:, 1]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(7.24551, 5.2), sharey="row")

plt.subplots_adjust(left=0.072, right=0.99, top=0.958, bottom=0.07, hspace=0.2, wspace=0.05)
axes = axes.ravel()

# model line
for ix, svr in enumerate(svrs_new):
    print("SVR Test score:", svr.score(X, svr.fit(X, y).predict(X)))
    SVR_all = np.stack([X[:, 0], svr.fit(X, y).predict(X)], 1)
    SVR_all = SVR_all[SVR_all[:, 0].argsort()]
    rev_transform = OC.pca.inverse_transform(SVR_all)

    # first row
    # model line
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=colors[ix],
        lw=lw,
        label="model",
    )
    # support vector scatter
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor=colors[ix],
        edgecolor="none",
        alpha=0.7, marker=".",
        s=40,
        label="SV ({})".format(len(X[svr.support_]))
    )
    # other datapoints scatter

    otherX = X[np.setdiff1d(np.arange(len(X)), svr.support_)]
    otherY = y[np.setdiff1d(np.arange(len(X)), svr.support_)]
    if len(otherX) > 1:
        PC_density_x, PC_density_y, PC_kwargs = CMD_density_design(
            [otherX.reshape(len(otherX), ), otherY.reshape(len(otherY), )], to_RBG=[0.27, 0.27, 0.27],
            from_RBG=[0.74, 0.74, 0.74], density_plot=False)
        PC_kwargs["s"] = 40

    axes[ix].scatter(
        PC_density_x, PC_density_y,
        # X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        # y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        # facecolor="k",
        # edgecolor="none",
        # alpha=0.5, marker = ".",
        # s=50,
        # label="other data ({})".format(int(len(X) - len(X[svr.support_]))),
        **PC_kwargs)
    # legend
    ymin, ymax = axes[0].get_ylim()
    axes[ix].set_ylim(ymin, ymax)
    axes[ix].legend(
        loc="upper left",
    )

    # Second row
    # model line
    axes[ix + 4].plot(
        rev_transform[:, 0],
        rev_transform[:, 1],
        color=colors[ix],
        lw=lw,
        label="model",
        # label="{} model".format(kernel_label[ix]),
    )
    OC_kwargs["s"] = 40
    # other datapoints
    axes[ix + 4].scatter(OC_density_x,
                         OC_density_y,
                         label="data", **OC_kwargs)
    # legend
    if ix == 0:
        ymin2, ymax2 = axes[4].get_ylim()
    axes[ix + 4].set_ylim(ymax2, ymin2)
    axes[ix + 4].legend(
        loc="upper right",
    )

axes[1].set_xlabel("PCA X", x=1., labelpad=0)
axes[0].set_ylabel("PCA Y", labelpad=0)

axes[4].set_ylabel(r"M$_{\mathrm{G}}$", labelpad=8)
axes[5].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} -\mathrm{G}_{\mathrm{RP}}$", x=1., labelpad=0)

axes[0].set_title("Default")  # \n (C=1, gamma = 'scale', epsilon = 0.1)")
axes[1].set_title("GridSearchCV")  # \n (C=100, gamma = 0.1, epsilon = 0.01)")
axes[2].set_title("HalvingGridSearchCV")  # \n (C=10, gamma = 0.05, epsilon = 0.0114)")
axes[3].set_title("BayesSearchCV")  # \n (C=10, gamma = 0.05, epsilon = 0.0114)")
# plt.suptitle("Example {}: default vs. tuned hyperparameters".format(OC.name), fontsize=16)

plt.show()
if save_plot:
    fig.savefig(output_path + "HP_Gridsearches_{}.pdf".format(OC.name), dpi=600)
