import os
from datetime import date

from pre_processing import *
from version_I.Confidence_intervals import *
from version_I.old_classfile import *
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
ScoCen_seb = "data/Cluster_data_raw/Sco-Cen-young_Seb_X_EDR3.csv"

df_columns = ["cluster_name", "parallax", "parallax_error", "phot_g_mean_mag_corrected", "phot_bp_mean_mag_DR3",
              "phot_rp_mean_mag_DR3", "phot_g_mean_mag_error_corrected", "phot_bp_mean_mag_error",
              "phot_rp_mean_mag_error", "bp_rp_DR3",
              "ruwe_DR3", "phot_bp_rp_excess_factor_corrected", "stability", "age_lts", "age_t-dist"]

new_colnames = ["Cluster_id", "plx", "e_plx", "G_mag", "BP_mag", "RP_mag", "e_G_mag", "e_BP_mag", "e_RP_mag", "BP_RP",
                "ruwe", "bp_rp_excess_factor", "stability", "age_lts", "age_tdist"]

ScoCen_clusters, ScoCen_df = create_df(ScoCen_seb, columns=df_columns, names=new_colnames, name_dict=None,
                                       label_col=None, excess=0.025)
#                                      stab=0.5, ruwe = 1.4, excess= 0.03)
# ----------------------------------------------------------------------------------------------------------------------
CI_raw = "data/Cluster_data_raw/Cantat-Gaudin_Xmatch_GAIA.csv"

CI_cols = ["Cluster", "parallax_EDR3", "parallax_error_EDR3", "phot_g_mean_mag_corrected", "phot_bp_mean_mag_EDR3",
           "phot_rp_mean_mag_EDR3", "phot_g_mean_mag_error_corrected", "phot_bp_mean_mag_error",
           "phot_rp_mean_mag_error",
           "bp_rp_EDR3"]

CI_names = ["Cluster_id", "plx", "e_plx", "G_mag", "BP_mag", "RP_mag", "e_G_mag", "e_BP_mag", "e_RP_mag", "BP_RP"]

CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names)
# ----------------------------------------------------------------------------------------------------------------------


Archive_clusters = np.concatenate([CI_clusters,ScoCen_clusters], axis = 0)
Archive_df = pd.concat([CI_df,ScoCen_df], axis = 0)

preprocess = True
sns.set_style("darkgrid")
Fig, vax2 = plt.subplots(figsize=(20, 20), nrows=3, ncols=6)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

iso_array = []
ages_LTS = []
ages_tdist = []
N_stars = []
cluster_w_stars = []

for i, ax2 in enumerate(Archive_clusters):
    OC = star_cluster(Archive_clusters[i], Archive_df)


    if OC.Nstars > 98:
        cluster_w_stars.append(OC)
        ages_LTS.append(OC.data["age_lts"].unique()[0])
        ages_tdist.append(OC.data["age_tdist"].unique()[0])
        N_stars.append(OC.Nstars)
        OC.kwargs_CMD["s"] = 50


        #OC = cluster_w_stars[i]

        pca = PCA(n_components=2)
        pca_arr = pca.fit_transform(OC.CMD)

        evals = np.logspace(-2, -1.5, 20)
        #gvals = np.logspace(-4, -1, 50)
        Cvals = np.logspace(-2, 2, 20)

        param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals,
                             epsilon=evals)

        params = SVR_Hyperparameter_tuning(pca_arr, param_grid)

        svr = SVR(**params)

        svr_predict = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)

        X = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)
        Y = pca_arr[:, 1]

        Y_all = svr.fit(X, Y).predict(svr_predict)
        print("SVR Test score:", svr.score(svr_predict, Y.ravel()))

        SVR_all = np.stack([svr_predict[:, 0], Y_all], 1)
        SVR_all = SVR_all[SVR_all[:, 0].argsort()]
        rev_transform = pca.inverse_transform(SVR_all)
        iso_array.append(rev_transform)


for i, ax2 in enumerate(vax2.flat[:]):
    try:

        OC = cluster_w_stars[i]

        kr = int(0.05 * len(iso_array[i][:, 0]))
        #kr = 0
        # 6. Plot the result for each cluster
        # Fig = plt.figure(figsize=(10, 8))
        sns.set_style("darkgrid")
        sns.set(font_scale=1.5)

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

        OC.kwargs_CMD["s"] = 50
        ax2.scatter(OC.density_x, OC.density_y, label="Sources", **OC.kwargs_CMD)
        ax2.plot(iso_array[i][kr:, 0], iso_array[i][kr:, 1], color="red")
        # ax2.set_ylabel(r"M$_{\rm G}$")

        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymax, ymin)
        for i in [0, 6, 12, 18]:
            ax2.set_ylabel(r"M$_{\rm G}$")
        for i in range(18, 24):
            ax2.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
        # ax2.legend(loc="best", fontsize=16)
        ax2.set_title(OC.name.replace("_", " "), y=0.97)

    except IndexError:
        pass

    # plt.subplots_adjust(top=0.92, left=0.1, right=0.95, bottom=0.12)

Fig.show()
#Fig.savefig(output_path + "composite_N70_new_grid.pdf", dpi=500)


Fig2 = plt.figure(figsize=(7, 5))

ax1 = plt.subplot2grid((1, 1), (0, 0))
#ax2 = plt.subplot2grid((1, 2), (0, 1))
palette = sns.color_palette("rocket",len(cluster_w_stars))#
palette_cmap = sns.color_palette("rocket_r",as_cmap=True)#

idx_es = np.arange(len(ages_tdist))
id_ages = np.stack([idx_es,ages_tdist],axis = 1)

sorted_ages = id_ages[(-id_ages[:,1]).argsort()]
print(sorted_ages)

for k,element in enumerate(sorted_ages[:,1]):
#for clus in ScoCen_clusters:
    #print(element)
    cluster_id = int(sorted_ages[k,0])
    CMD_isochrone = iso_array[cluster_id]
    kr = int(0.05 * len(CMD_isochrone[:, 0]))
    ax1.plot(CMD_isochrone[kr:, 0], CMD_isochrone[kr:, 1], color = palette[k],  lw=0.5, label= ScoCen_clusters[k])

ymin, ymax = ax1.get_ylim()
ax1.set_ylim(ymax, ymin)
ax1.set_xlabel(r"G$_{\rm BP}$ - G$_{\rm RP}$")
ax1.set_ylabel(r"M$_{\rm G}$")
#Fig.subplots_adjust(right=0.8, wspace=0.2)
#cbar_ax = Fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap=palette_cmap,  norm=plt.Normalize(vmin=np.min(sorted_ages[:,1]), vmax=np.max(sorted_ages[:,1])))
plt.colorbar(sm)
Fig2.show()
Fig2.savefig(output_path+"test_Sco-Cen.pdf", dpi = 500)

#------------------------------------------------

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