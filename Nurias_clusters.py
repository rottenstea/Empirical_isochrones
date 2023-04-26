import os
from datetime import date
from Classfile import *
from pre_processing import case_study_names, case_study_dfs
import seaborn as sns
import matplotlib.pyplot as plt

"""
This is a file that plays around with the cluster selection of Nuria. It is fully functional, but holds only one of
the different plot combinations that I have since tried with the panSTARRs and 2MASS data.
"""

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

HP_file = "data/Hyperparameters/Case_studies_with_errors.csv"
try:
    pd.read_csv(HP_file)
except FileNotFoundError:
    with open(HP_file, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

# ----------------------------------------------------------------------------------------------------------------------
Pleiades_cluster, Pleiades_df = case_study_names[0], case_study_dfs[0]

Pleiades_filtered_df = Pleiades_df[Pleiades_df["imag"] > 13]
# ----------------------------------------------------------------------------------------------------------------------
J, H, K = np.genfromtxt("data/Isochrones/PARSEC_isochrones/Nuria_clusters/2MASS_30Myr.txt", usecols=(-3, -2, -1), unpack=True)
i = np.genfromtxt("data/Isochrones/PARSEC_isochrones/Nuria_clusters/panSTARRs1_30Myr.txt", usecols=(-4))

IC4665_cluster, IC4665_df = case_study_names[1], case_study_dfs[1]

IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13)]

N_df = pd.concat([Pleiades_filtered_df, IC4665_filtered_df], axis=0)
# ----------------------------------------------------------------------------------------------------------------------
sns.set_style("darkgrid")
colors = ["red", "darkorange"]
kwargs = dict(grid=None, HP_file=HP_file)
save_plot = False

fig = plt.figure(figsize=(4, 6))
ax = plt.subplot2grid((1, 1), (0, 0))

cm = plt.cm.get_cmap("crest")

for i, cluster in enumerate(case_study_names[:]):

    OC = star_cluster(cluster, N_df, catalog_mode=False)
    deltas = OC.create_CMD(CMD_params=["imag", "imag", "zmag"], return_errors=True)

    # 3. Do some initial HP tuning if necessary
    try:
        params = OC.SVR_read_from_file(HP_file)
    except IndexError:
        curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

    # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
    n_boot = 1000
    result_df = OC.isochrone_and_intervals(n_boot=n_boot, output_loc="data/Isochrones/", kwargs=kwargs)

    # 5. Plot the result
    fig1 = plt.figure(figsize=(4, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    sc = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=OC.weights, cmap=cm, s=20)
    ax1.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
    ax1.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
    ax1.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")
    plt.colorbar(sc)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])
    ax1.set_xlabel(OC.CMD_specs["axes"][1])
    ax1.set_title(OC.name)

    fig1.show()
    if save_plot:
        fig1.savefig(output_path+"{}_isochrone_errormap.png".format(OC.name),dpi=500)

    error_fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    s1 = ax1.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[0], cmap=cm, marker=".", s=5)
    ax1.set_title("imag error (cax)")
    plt.colorbar(s1, ax=ax1)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymax, ymin)
    ax1.set_ylabel(OC.CMD_specs["axes"][0])
    ax1.set_xlabel(OC.CMD_specs["axes"][1])

    s2 = ax2.scatter(OC.CMD[:, 0], OC.CMD[:, 1], label=OC.name, c=deltas[1], cmap=cm, marker=".", s=5)
    ax2.set_title("Kmag error (cax)")
    plt.colorbar(s2, ax=ax2)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymax, ymin)
    ax2.set_ylabel(OC.CMD_specs["axes"][0])
    ax2.set_xlabel(OC.CMD_specs["axes"][1])

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
    ax4.set_ylabel(OC.CMD_specs["axes"][0])
    ax4.set_xlabel(OC.CMD_specs["axes"][1])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    error_fig.show()

    if save_plot:
        error_fig.savefig(output_path+"{}_errorplot.png".format(OC.name), dpi=500)

    if i == 0:
        ax.scatter(OC.density_x, OC.density_y, label="IC_4665 data", **OC.kwargs_CMD)
    elif i == 1:
        OC_density_x, OC_density_y, OC_kwargs = CMD_density_design([OC.CMD[:, 0], OC.CMD[:, 1]],
                                                                   to_RBG=[0.06, 0.55, 0.44],
                                                                   from_RBG=[0.13, 0.75, 0.62], density_plot=False)
        ax.scatter(OC_density_x, OC_density_y, label="Pleiades data", **OC_kwargs)
    ax.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc", alpha=0.7)
    ax.plot(result_df["m_x"], result_df["m_y"], color=colors[i], label="Isochrone")
    ax.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc", alpha=0.7)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)
ax.set_ylabel(r"abs mag i")
ax.set_xlabel(r"${\rm i}$ - ${\rm K}$")
ax.set_title("Empirical isochrones")
fig.show()

if save_plot:
    fig.savefig(output_path + "Comparison_Pleiades_IC4665_data.png", dpi=500)

