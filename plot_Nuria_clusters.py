import os
from datetime import date

import pandas as pd

from Classfile import *
from pre_processing import case_study_names, case_study_dfs
import seaborn as sns
import matplotlib.pyplot as plt

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

data_path = "/Users/alena/PycharmProjects/PaperI/"
HP_file_cs = data_path + "data/Hyperparameters/Case_studies_with_errors.csv"
try:
    pd.read_csv(HP_file_cs)
except FileNotFoundError:
    with open(HP_file_cs, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

bhac15 = data_path + "data/Isochrones/BHAC15/baraffe15.csv"

sns.set_style("darkgrid")
colors = ["red", "darkorange"]
kwargs = dict(grid=None, HP_file=HP_file_cs)
save_plot = False

# Pleiades
Pleiades_cluster, Pleiades_df = case_study_names[0], case_study_dfs[0]
Pleiades_filtered_df = Pleiades_df[Pleiades_df["imag"] > 13]

# IC 4665
IC4665_cluster, IC4665_df = case_study_names[1], case_study_dfs[1]
IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13)]
N_df = pd.concat([Pleiades_filtered_df, IC4665_filtered_df], axis=0)

# Test of the Bhac15 isochrones on Pleiades GAIA
HP_CII = data_path+"data/Hyperparameters/CatalogII.csv"
try:
    pd.read_csv(HP_CII)
except FileNotFoundError:
    with open(HP_CII, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

from pre_processing import CII_df
kwargs_CII = dict(grid=None, HP_file=HP_CII)

# store cluster objects for later
OCs = []
for i, cluster in enumerate(case_study_names[:]):

    OC = star_cluster(cluster, N_df, catalog_mode=False)
    deltas = OC.create_CMD(CMD_params=["imag", "imag", "Kmag"], return_errors=True)

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file_cs)
    except IndexError:
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 100
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, parallel_jobs = 10)

    setattr(OC,"isochrone",result_df[["m_x","m_y"]])
    setattr(OC,"upper",result_df[["u_x","u_y"]])
    setattr(OC,"lower",result_df[["l_x","l_y"]])

    OCs.append(OC)

cm = plt.cm.get_cmap("crest")

for OC in OCs:
    fig1 = plt.figure(figsize=(4, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    sc = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, s=20)
    ax1.plot(OC.lower["l_x"], OC.lower["l_y"], color="grey", label="5. perc")
    ax1.plot(OC.isochrone["m_x"], OC.isochrone["m_y"], color="red", label="Isochrone")
    ax1.plot(OC.upper["u_x"], OC.upper["u_y"], color="grey", label="95. perc")
    plt.colorbar(sc)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])
    ax1.set_xlabel(OC.CMD_specs["axes"][1])
    ax1.set_title(OC.name)

    plt.show()
    # if save_plot:
    #     fig1.savefig(output_path+"{}_isochrone_errormap.png".format(OC.name),dpi=500)


for OC in OCs:

    deltas = OC.create_CMD(CMD_params=["imag", "imag", "Kmag"], return_errors=True)

    error_fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1), sharey = ax1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), sharex = ax1 )
    ax4 = plt.subplot2grid((2, 2), (1, 1), sharey = ax3)

    s1 = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[0], cmap=cm, marker=".", s=5)
    ax1.set_title("imag error (cax)")
    plt.colorbar(s1, ax=ax1)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])

    s2 = ax2.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[1], cmap=cm, marker=".", s=5)
    ax2.set_title("Kmag error (cax)")
    plt.colorbar(s2, ax=ax2)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymax, ymin)

    cax_error = np.sqrt(deltas[0] ** 2 + deltas[1] ** 2)
    s3 = ax3.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=cax_error, cmap=cm, marker=".", s=5)
    ax3.set_title("cax errors")
    plt.colorbar(s3, ax=ax3)
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymax, ymin)
    ax3.set_ylabel(OC.CMD_specs["axes"][0])
    ax3.set_xlabel(OC.CMD_specs["axes"][1])

    s4 = ax4.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, marker=".", s=5)
    ax4.set_title("weights")
    plt.colorbar(s4, ax=ax4)
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(ymax, ymin)
    ax4.set_xlabel(OC.CMD_specs["axes"][1])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()

bhac15_df = pd.read_csv(bhac15)
Pleiades_age = [0.08,0.1,0.12]
IC4665_age = [0.025,0.03,0.04]

isos_IC4665 = [bhac15_df[bhac15_df["Age_p1"]== i] for i in IC4665_age ]
isos_Pleiades = [bhac15_df[bhac15_df["Age_p1"]== i] for i in Pleiades_age ]


# Pleiades Plot
OC = OCs[0]
fig = plt.figure(figsize=(4,6))
plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
for j,isos in enumerate(isos_Pleiades):
    plt.plot(isos["i_p1"]-isos["Mk"], isos["i_p1"], label = "{} Myr".format(int(Pleiades_age[j]* 10**3)))

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name)

parsec_df_1 = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Isochrones/PARSEC_isochrones/PARSEC+COLIBRI_30Myr_GDR2-Evans_2MASS_ps1.csv")

parsec_df = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Isochrones/PARSEC_isochrones/2MASS_GAIA_PS_30_Myr.csv")
parsec_df.head()

btsettl_df = pd.read_csv("/Users/alena/PycharmProjects/PaperI/data/Isochrones/PARSEC_isochrones/Nuria_clusters/BTSettl_30Myr_GDR2_ps1_2mass.csv")
btsettl_df.head()


# IC 4665 Plot
OC = OCs[1]
fig = plt.figure(figsize=(5,6))
plt.scatter(OC.density_x, OC.density_y+ 5*np.log10(1000/OC.data.plx[0])-5 , s = 10, color = "black")

for j,isos in enumerate(isos_IC4665[:]):
    if j==1:
        plt.plot(isos["i_p1"]-isos["Mk"] + 0.39644, isos["i_p1"]+ 5*np.log10(1000/OC.data.plx[0])-5 + 0.4526, color = "orange", label = "BHAC15 {} Myr".format(int(IC4665_age[j]* 10**3)))
a=100
plt.plot(parsec_df["iP1mag"]-parsec_df["Ksmag"]+ 0.39644, parsec_df["iP1mag"]+ 5*np.log10(1000/OC.data.plx[0])-5+ 0.4526,color="red", label ="PARSEC 30 Myr (OBC)")
plt.plot(parsec_df_1["iP1mag"]-parsec_df_1["Ksmag"]+ 0.39644, parsec_df_1["iP1mag"]+ 5*np.log10(1000/OC.data.plx[0])-5+ 0.4526,color="firebrick", label ="PARSEC 30 Myr NURIA")
plt.plot(btsettl_df["i_p1"]-btsettl_df["K"]+ 0.39644, btsettl_df["i_p1"]+ 5*np.log10(1000/OC.data.plx[0])-5+ 0.4526, color = "violet", label = "BT-Settl 30 Myr")

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name + " with extinction")
plt.ylim(25,12)
plt.xlim(1.75,7)
plt.ylabel("$i$")
plt.xlabel("$i$-$K$")

plt.show()
#fig.savefig(output_path+"IC_4665_theoretical_isochrones_extinction.png", dpi=500)



OC = star_cluster("Melotte_22", CII_df)
OC.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])

try:
    params = OC.SVR_read_from_file(HP_CII)
except IndexError:
    curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

n_boot = 1000
result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs = kwargs)


# Plot
isos_Pleiades = [bhac15_df[bhac15_df["Age_GAIA"]== i] for i in Pleiades_age ]

fig = plt.figure(figsize=(5,6))

plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)

for j,isos in enumerate(isos_Pleiades):
    plt.plot(isos["G_BP"]-isos["G_RP"], isos["G"], label = "{} Myr".format(int(Pleiades_age[j]* 10**3)))

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name)
plt.show()

# G- RP CMD
OC.create_CMD(CMD_params=["Gmag", "Gmag", "RPmag"])
try:
    params = OC.SVR_read_from_file(HP_CII)
except IndexError:
    curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)
n_boot = 1000
result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs = kwargs)

# Plot
isos_Pleiades = [bhac15_df[bhac15_df["Age_GAIA"]== i] for i in Pleiades_age ]

fig = plt.figure(figsize=(5,6))
plt.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)

for j,isos in enumerate(isos_Pleiades):
    plt.plot(isos["G"]-isos["G_RP"], isos["G"], label = "{} Myr".format(int(Pleiades_age[j]* 10**3)))

plt.gca().invert_yaxis()
plt.legend(loc="upper right")
plt.title(OC.name)


# Passband combinations
CMD_combis = [["rmag", "rmag", "imag"], ["imag", "imag", "zmag"], ["imag", "imag", "ymag"],
              ["imag", "imag", "Kmag"], ["ymag", "ymag", "Kmag"], ["Jmag", "Jmag", "Kmag"]]


from_color = [[0.74, 0.74, 0.74],[0.62, 0.79, 0.88],[0.72,0.78,0.71]]
to_color =[[0.27, 0.27, 0.27],[0.0, 0.25, 0.53],[0.17,0.36,0.25]]

fig1 = plt.figure(figsize=(15, 15))
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1))
ax3 = plt.subplot2grid((2, 3), (0, 2))
ax4 = plt.subplot2grid((2, 3), (1, 0))
ax5 = plt.subplot2grid((2, 3), (1, 1))
ax6 = plt.subplot2grid((2, 3), (1, 2))

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

fig2 = plt.figure(figsize=(15, 15))
ax11 = plt.subplot2grid((2, 3), (0, 0))
ax22 = plt.subplot2grid((2, 3), (0, 1))
ax33 = plt.subplot2grid((2, 3), (0, 2))
ax44 = plt.subplot2grid((2, 3), (1, 0))
ax55 = plt.subplot2grid((2, 3), (1, 1))
ax66 = plt.subplot2grid((2, 3), (1, 2))

axes_2 = [ax11, ax22, ax33, ax44, ax55, ax66]


kwargs_new = dict(HP_file=HP_file_cs, grid=None)

for k,filters in enumerate(CMD_combis[:]):
    for i, cluster in enumerate(case_study_names[:]):

        OC = star_cluster(cluster, N_df, catalog_mode=False)
        OC.create_CMD(CMD_params=filters)
        #filtered_CMD = OC.CMD[OC.CMD[:,1] > cuts[i][k]]

        try:
            params = OC.SVR_read_from_file(HP_file_cs)
        except IndexError:
            curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs_new)

        n_boot = 100
        result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs_new)

        OC_density_x, OC_density_y, OC_kwargs = CMD_density_design(OC.CMD, to_RBG=to_color[i], from_RBG=from_color[i], cluster_obj=OC, density_plot=False)
        axes[k].scatter(OC_density_x, OC_density_y, label="{} data".format(OC.name), **OC_kwargs)
        axes[k].plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc", alpha=0.7)
        axes[k].plot(result_df["m_x"], result_df["m_y"], color=colors[i], label="Isochrone")
        axes[k].plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc", alpha=0.7)


        axes[k].set_ylabel(OC.CMD_specs["axes"][0])
        axes[k].set_xlabel(OC.CMD_specs["axes"][1])


        axes_2[k].plot(result_df["m_x"], result_df["m_y"], color=colors[i], label="Isochrone")

        axes_2[k].set_ylabel(OC.CMD_specs["axes"][0])
        axes_2[k].set_xlabel(OC.CMD_specs["axes"][1])


plt.subplots_adjust(hspace=0.25)


axes[0].set_xlim(-0.5,3.5)
axes[1].set_xlim(-0.,2.5)
axes[2].set_xlim(-0.,4)
axes[3].set_xlim(1.5,7.2)
axes[4].set_xlim(1,4)
axes[5].set_xlim(0,2)

axes[0].set_ylim(19,3)
axes[1].set_ylim(19.,5)
axes[2].set_ylim(18,5)
axes[3].set_ylim(18,5)
axes[4].set_ylim(15,4)
axes[5].set_ylim(13,2)

fig1.savefig(output_path+"All_filters_data.png", dpi = 500)
fig1.suptitle("Empirical isochrones", fontsize = 16, y = 0.9)
plt.legend(bbox_to_anchor=(-1,0.15,1, 1), loc="upper right", ncol =3, fontsize = 13)


axes_2[0].set_xlim(-0.5,3.5)
axes_2[1].set_xlim(-0.,2.5)
axes_2[2].set_xlim(-0.,4)
axes_2[3].set_xlim(1.5,7.2)
axes_2[4].set_xlim(1,4)
axes_2[5].set_xlim(0,2)

axes_2[0].set_ylim(19,3)
axes_2[1].set_ylim(19.,5)
axes_2[2].set_ylim(18,5)
axes_2[3].set_ylim(18,5)
axes_2[4].set_ylim(15,4)
axes_2[5].set_ylim(13,2)



fig2.suptitle("Empirical isochrones", fontsize = 16, y = 0.9)
plt.show()
#fig2.savefig(output_path+"All_filters_only_isochrones.png", dpi = 500)








