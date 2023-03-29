import os
from datetime import date

from pre_processing import *
# from version_I.Confidence_intervals import *
from version_I.Support_Vector_Regression import SVR_Hyperparameter_tuning
from Classfile import *
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from astropy.io import fits

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

# Catalog import
# ----------------------------------------------------------------------------------------------------------------------
CIII_raw = "data/Cluster_data/all_ages/CatalogIII_DR3_Seb_ages.csv"

CIII_cols = ["cluster_name", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "logage_lts", "logage_tdist",
             "ruwe", "fidelity_v2", "stability", "G_err", "G_BPerr", "G_RPerr"]

CIII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
              "BP-G", "G-RP",
              "age_lts", "age_tdist", "ruwe", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"]

q_filter = {"parameter": ["ruwe", "plx", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"],
            "limit": ["upper", "lower", "lower", "lower", "upper", "upper", "upper"],
            "value": [1.4, 0, 0.5, 6, 0.007, 0.05, 0.03]}

CIII_clusters, CIII_df = create_df(CIII_raw, CIII_cols, CIII_names, q_filter)
# ----------------------------------------------------------------------------------------------------------------------
CI_raw = "data/Cluster_data/all_ages/CatalogI_BCD_ages.csv"

CI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
           "G-RP",
           "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
           "RUWE", "Proba"]

CI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability"]

q_filter = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"], "value": [1.4, 0, 0.49]}

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter)

CI_df["ref_age"] = CI_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# Coma Ber (Melotte 111) == ADD-ON I
# ----------------------------------------------------------------------------------------------------------------------

AOI_raw = "data/Cluster_data/all_ages/Coma_Ber_CD_ages.csv"

AOI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
            "RUWE"]

AOI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

AOI_clusters, AOI_df = create_df(AOI_raw, AOI_cols, AOI_names, q_filter)

AOI_df["ref_age"] = AOI_df["age_C"]

# ----------------------------------------------------------------------------------------------------------------------
# Hyades (Melotte 25) == ADD-ON II
# ----------------------------------------------------------------------------------------------------------------------

AOII_raw = "data/Cluster_data/all_ages/Hyades_CD_ages.csv"

AOII_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
             "G-RP",
             "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
             "RUWE"]

AOII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
              "BP-G", "G-RP",
              "age_C", "av_C", "age_D", "av_D", "ruwe"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

AOII_clusters, AOII_df = create_df(AOII_raw, AOII_cols, AOII_names, q_filter)
AOII_df["ref_age"] = AOII_df["age_C"]
# ----------------------------------------------------------------------------------------------------------------------
# Meingast 1 == CASE STUDY I
# ----------------------------------------------------------------------------------------------------------------------

CSI_raw = "data/Cluster_data/all_ages/Meingast1_stab_24_CuESSIV_ages.csv"

CSI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
            "G-RP",
            "logage_Curtis", "logage_ESSIV",
            "RUWE", "Stab"]

CSI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "age_Cu", "age_ESSIV", "ruwe", "stability"]

q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

CSI_clusters, CSI_df = create_df(CSI_raw, CSI_cols, CSI_names, q_filter)

CSI_df["ref_age"] = CSI_df["age_Cu"]

Archive_clusters = np.concatenate([CI_clusters,AOI_clusters, AOII_clusters, CSI_clusters], axis = 0)
Archive_df = pd.concat([CI_df,AOI_df, AOII_df, CSI_df], axis = 0)

preprocess = True
sns.set_style("darkgrid")
Fig, vax2 = plt.subplots(figsize=(12, 23), nrows=10, ncols=7)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

iso_array = []
ages_LTS = []
ages_tdist = []
N_stars = []
cluster_w_stars = []
j = 0

params_all_CIII = [
    {'C': 0.016237767391887217, 'epsilon': 0.011288378916846888, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.18329807108324356, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.06951927961775606, 'epsilon': 0.023357214690901226, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.18329807108324356, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.026366508987303583, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.011993539462092343, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 37.92690190732246, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.29763514416313175, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 23.357214690901213, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 3.359818286283781, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 2.06913808111479, 'epsilon': 0.0206913808111479, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.01438449888287663, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.18329807108324356, 'epsilon': 0.018329807108324356, 'gamma': 'scale', 'kernel': 'rbf'}
]

params_all_CI = [
    {'C': 2.06913808111479, 'epsilon': 0.021983926488622893, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 3.359818286283781, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.4832930238571752, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 37.92690190732246, 'epsilon': 0.016237767391887217, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 37.92690190732246, 'epsilon': 0.01438449888287663, 'gamma': 'scale', 'kernel': 'rbf'},  # 5
    {'C': 2.06913808111479, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 8.858667904100823, 'epsilon': 0.010624678308940415, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 2.06913808111479, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 3.359818286283781, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},  # 10
    {'C': 1.2742749857031335, 'epsilon': 0.012742749857031334, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.021983926488622893, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 8.858667904100823, 'epsilon': 0.023357214690901226, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 8.858667904100823, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 23.357214690901213, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},  # 15
    {'C': 100.0, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.7847599703514611, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 61.584821106602604, 'epsilon': 0.011993539462092343, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 23.357214690901213, 'epsilon': 0.017252105499420408, 'gamma': 'scale', 'kernel': 'rbf'},  # 20
    {'C': 2.06913808111479, 'epsilon': 0.010624678308940415, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.01353876180022544, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.026366508987303583, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},  # 25
    {'C': 0.016237767391887217, 'epsilon': 0.01353876180022544, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.026366508987303583, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 37.92690190732246, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 61.584821106602604, 'epsilon': 0.026366508987303583, 'gamma': 'scale', 'kernel': 'rbf'},  # 30
    {'C': 2.06913808111479, 'epsilon': 0.026366508987303583, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 1.2742749857031335, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.7847599703514611, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},  # 35
    {'C': 2.06913808111479, 'epsilon': 0.021983926488622893, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 37.92690190732246, 'epsilon': 0.0206913808111479, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.7847599703514611, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},  # 40
    {'C': 23.357214690901213, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 1.2742749857031335, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 61.584821106602604, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 2.06913808111479, 'epsilon': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.04281332398719394, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},  # 45
    {'C': 14.38449888287663, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 1.2742749857031335, 'epsilon': 0.0206913808111479, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 1.2742749857031335, 'epsilon': 0.023357214690901226, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},  # 50
    {'C': 2.06913808111479, 'epsilon': 0.016237767391887217, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 2.06913808111479, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 1.2742749857031335, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.4832930238571752, 'epsilon': 0.01353876180022544, 'gamma': 'scale', 'kernel': 'rbf'},  # 55
    {'C': 37.92690190732246, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 23.357214690901213, 'epsilon': 0.011993539462092343, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.018329807108324356, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.4832930238571752, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 14.38449888287663, 'epsilon': 0.01947483039908756, 'gamma': 'scale', 'kernel': 'rbf'},  # 60
    {'C': 0.18329807108324356, 'epsilon': 0.026366508987303583, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 5.455594781168514, 'epsilon': 0.029763514416313176, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.18329807108324356, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 8.858667904100823, 'epsilon': 0.026366508987303583, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 2.06913808111479, 'epsilon': 0.01438449888287663, 'gamma': 'scale', 'kernel': 'rbf'},  # 65
    {'C': 5.455594781168514, 'epsilon': 0.021983926488622893, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 100.0, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'},
]

params_all_Archive = params_all_CI + [{'C': 37.92690190732246, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}, # Coma Ber
                                      {'C': 0.04281332398719394, 'epsilon': 0.018329807108324356, 'gamma': 'scale', 'kernel': 'rbf'}, #Hyades
                                      {'C': 2.06913808111479, 'epsilon': 0.03162277660168379, 'gamma': 'scale', 'kernel': 'rbf'} # Meingast 1
                                      ]

ages_archive = []


#------------------
# SCALED WEIGHTS
#------------------

import sklearn.preprocessing as p
mmscaler = p.MinMaxScaler()

for i, cluster in enumerate(Archive_clusters[:]):
    OC = star_cluster(cluster, Archive_df)

    if OC.Nstars > 90:
        cluster_w_stars.append(OC)
        if not np.isnan(OC.data["ref_age"].unique()[0]):
            ages_archive.append(OC.data["ref_age"].unique()[0])
        else:
            if not np.isnan(OC.data["age_D"].unique()[0]):
                ages_archive.append(OC.data["age_D"].unique()[0])
            else:
                print(OC.name)
                ages_archive.append(0)
        # ages_LTS.append(OC.data["age_lts"].unique()[0])
        # ages_tdist.append(OC.data["age_tdist"].unique()[0])
        N_stars.append(OC.Nstars)

        OC.create_CMD()

        w = OC.weights.reshape(len(OC.weights),1)
        normalized_weights = mmscaler.fit_transform(w).reshape(len(w),)
        #print(OC.weights)

        # OC = cluster_w_stars[i]

        #pca = PCA(n_components=2)
        #pca_arr = pca.fit_transform(OC.CMD)
        pca_arr = OC.PCA_XY

        evals = np.logspace(-2, -1.5, 20)
        # gvals = np.logspace(-4, -1, 50)
        Cvals = np.logspace(-2, 2, 20)

        param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals,
                          epsilon=evals)

        params = SVR_Hyperparameter_tuning(pca_arr, param_grid, weight_data=normalized_weights)
        #params = params_all_Archive[j]

        svr = SVR(**params)

        svr_predict = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)

        X = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)
        Y = pca_arr[:, 1]

        Y_all = svr.fit(X, Y).predict(svr_predict)
        print("SVR Test score:", svr.score(svr_predict, Y.ravel()))

        SVR_all = np.stack([svr_predict[:, 0], Y_all], 1)
        SVR_all = SVR_all[SVR_all[:, 0].argsort()]
        rev_transform = OC.pca.inverse_transform(SVR_all)
        iso_array.append(rev_transform)

        # f = CMD_density_design(OC.CMD, cluster_obj=OC)
        # OC.kwargs_CMD["s"] = 50
        # plt.plot(rev_transform[:, 0], rev_transform[:, 1], color="red")
        # f.show()
        # f.savefig(output_path + "{}_weights.pdf".format(OC.name), dpi=500)
        # print(j)
        # j += 1

for i, ax in enumerate(vax2.flat[:]):
    try:

        OC = cluster_w_stars[i]
        kr = 0

        sns.set_style("darkgrid")

        cs = ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
        ax.plot(iso_array[i][kr:, 0], iso_array[i][kr:, 1], color="red")

        ax.set_title(OC.name.replace("_", " "))
        if i in range(0, 70, 7):
            ax.set_ylabel(f"absolute {OC.CMD_specs['axes'][0]}")

        if i in range(63, 70):
            ax.set_xlabel(OC.CMD_specs["axes"][1])

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)

        plt.colorbar(cs, ax=ax)

        # PCA
        # ax1 = plt.subplot2grid((1, 2), (0, 0))

        # OC.kwargs_CMD["s"] = 50
        # ax1.scatter(pca_arr[:,0], pca_arr[:,1], label="Sources", **OC.kwargs_CMD)
        # ax1.plot(SVR_all[kr:, 0], SVR_all[kr:, 1], color="red")
        # ax1.set_ylabel(r"y PCA")

        # ymin, ymax = ax1.get_ylim()
        # ax1.set_ylim(ymax, ymin)
        # ax1.set_xlabel(r"X PCA")
        # ax1.legend(loc="best", fontsize=16)

        # real
        # ax2 = plt.subplot2grid((1, 2), (0, 1))

    except IndexError:
        pass

    # plt.subplots_adjust(top=0.92, left=0.1, right=0.95, bottom=0.12)
#for ax in range(67, 72):
#    Fig.delaxes(vax2.flat[ax])

plt.subplots_adjust(wspace=0.5, hspace=0.4)

Fig.show()
Fig.savefig(output_path + "composite_Archive_OCPCAXY_normweights.pdf", dpi=500)

'''
Fig2 = plt.figure(figsize=(7, 5))

ax1 = plt.subplot2grid((1, 1), (0, 0))
#ax2 = plt.subplot2grid((1, 2), (0, 1))
palette = sns.color_palette("crest",len(ages_archive))#
palette_cmap = sns.color_palette("crest_r",as_cmap=True)#

idx_es = np.arange(len(ages_archive))

id_ages = np.stack([idx_es,ages_archive],axis = 1)

sorted_ages = id_ages[(-id_ages[:,1]).argsort()]

#print(sorted_ages)


for k,element in enumerate(sorted_ages[:-1,1]):
#for clus in ScoCen_clusters:
    #print(element)
    cluster_id = int(sorted_ages[k,0])
    CMD_isochrone = iso_array[cluster_id]
    #kr = int(0.05 * len(CMD_isochrone[:, 0]))
    kr = 0
    ax1.plot(CMD_isochrone[kr:, 0], CMD_isochrone[kr:, 1], color = palette[k],  lw=0.5, label= Archive_clusters[k])

ymin, ymax = ax1.get_ylim()
ax1.set_ylim(16, -3)
ax1.set_xlim(-0.5, 5)
ax1.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
ax1.set_ylabel(r"M$_{\rm G}$")
#Fig.subplots_adjust(right=0.8, wspace=0.2)
#cbar_ax = Fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap=palette_cmap,  norm=plt.Normalize(vmin=np.min(sorted_ages[:-1,1]), vmax=np.max(sorted_ages[:-1,1])))
plt.colorbar(sm)
Fig2.show()
Fig2.savefig(output_path+"Age_summary_Archive_OCPCAXY.pdf", dpi = 500)

#------------------------------------------------

'''
'''



Dias_ages = "data/Ages/Clusters_X_Dias.fits"

Dias_file = fits.open(Dias_ages)
Dias_data = Dias_file[1].data

Dias_names = Dias_data['Cluster_1']
Dias_logA  = Dias_data["logage"]





SVR_params_CI = C.SVR_read_from_file()
#SVR_params_CII = C.SVR_read_from_file(target_methods + "best_params_SVR_CII.txt")

MSTO = [8, 16, 24, 28, 29, 36, 38, 40, 41, 43, 44, 51, 52, 53, 56]

MSTO_CII = [4]

CMD_isochrones_B = []


n_obs = 1000

for q, name in enumerate(L_archive.names):
    if name in Dias_names:
        idx, mask = L_archive.mask(name)
        OC = C.single_cluster(idx, mask, **CI_kwargs)
        print(OC.name)

        if q not in MSTO:

                method_kwargs = dict(keys=SVR_params_CI, sample_name=OC.name, svr_predict=OC.create_CMD)
                CMD_isochrone = C.SVR_PCA_calculation(OC.create_CMD, **method_kwargs)
                lower, median, upper = np.loadtxt(
                    target_results + "Resampled_isos/02/{0}_stats_{1}.csv".format(OC.name, n_obs), delimiter=',',
                    unpack=True)
                CMD_isochrones_B.append(np.stack([CMD_isochrone[:,0],median],axis=1))


        else:

            method_kwargs = dict(keys=SVR_params_CI, sample_name=OC.name, svr_predict=OC.create_CMD)
            CMD_isochrone = C.SVR_PCA_calculation(OC.create_CMD, **method_kwargs)
            lower = np.loadtxt(target_results + "Resampled_isos/02/{0}_lower_{1}.csv".format(OC.name, n_obs),
                               delimiter=',')
            median = np.loadtxt(target_results + "Resampled_isos/02/{0}_median_{1}.csv".format(OC.name, n_obs),
                                delimiter=',')
            upper = np.loadtxt(target_results + "Resampled_isos/02/{0}_upper_{1}.csv".format(OC.name, n_obs),
                               delimiter=',')
            CMD_isochrones_B.append(median)


plt.rcParams.update({'xtick.labelsize': 22})
plt.rcParams.update({'ytick.labelsize': 22})

Fig = plt.figure(figsize=(7, 8))

ax1 = plt.subplot2grid((1, 1), (0, 0))
#ax2 = plt.subplot2grid((1, 2), (0, 1))
palette = sns.color_palette("rocket",len(Dias_names))#
palette_cmap = sns.color_palette("rocket_r",as_cmap=True)#


idx_es = np.arange(len(Dias_logA))
id_ages = np.stack([idx_es,Dias_logA],axis = 1)
sorted_ages = id_ages[(-id_ages[:,1]).argsort()]
#sorted_ages = sorted_ages[::-1]
#print(sorted_ages)

for k,element in enumerate(sorted_ages[:,1]):


    cluster_id = int(sorted_ages[k,0])
    CMD_isochrone = CMD_isochrones_B[cluster_id]
    ax1.plot(CMD_isochrone[:, 0], CMD_isochrone[:, 1], color = palette[k],  lw=0.5)
    print(element,Dias_names[cluster_id])
        #plt.show()

ymin, ymax = ax1.get_ylim()
ax1.set_ylim(ymax, ymin)
ax1.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$", fontsize = 22)
ax1.set_ylabel(r"M$_{\rm G}$", fontsize = 22)
#Fig.subplots_adjust(right=0.8, wspace=0.2)
#cbar_ax = Fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap=palette_cmap,  norm=plt.Normalize(vmin=np.min(sorted_ages[:,1]), vmax=np.max(sorted_ages[:,1])))
cbar = plt.colorbar(sm)
cbar.set_label(r"$\log$(age)", fontsize = 22)
cbar.ax.tick_params(labelsize=22)

#plt.legend(ncol = 3,bbox_to_anchor = (0.5,-0.5),loc="lower center")
Fig.show()
Fig.savefig(target_results+"Ages_Dias_n1000.pdf", dpi = 300)
'''
