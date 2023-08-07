import my_utility
from pre_processing import case_study_names, case_study_dfs
from Classfile import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

output_path = my_utility.set_output_path()
save_plot = True
# ----------------------------------------------------------------------------------------------------------------------

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

# ----------------------------------------------------------------------------------------------------------------------
# paths
data_path = "/Users/alena/PycharmProjects/PaperI/"
HP_file_cs = data_path + "data/Hyperparameters/Case_studies_with_errors.csv"
my_utility.setup_HP(HP_file_cs)
kwargs = dict(grid=None, HP_file=HP_file_cs, catalog_mode=False)

# Pleiades
Pleiades_cluster, Pleiades_df = case_study_names[0], case_study_dfs[0]
Pleiades_filtered_df = Pleiades_df[Pleiades_df["imag"] > 13]

# IC 4665
IC4665_cluster, IC4665_df = case_study_names[1], case_study_dfs[1]
IC4665_filtered_df = IC4665_df[(IC4665_df["imag"] > 13)]
N_df = pd.concat([Pleiades_filtered_df, IC4665_filtered_df], axis=0)

fig, ax = plt.subplots(2, 3, figsize=(7.24551, 6.5))
axes = ax.ravel()

plt.subplots_adjust(left=0.062, bottom=0.0652, top=0.935, right=0.99, wspace=0.23, hspace=0.18)

# Passband combinations
CMD_combis = [["rmag", "rmag", "imag"], ["imag", "imag", "zmag"], ["imag", "imag", "ymag"],
              ["imag", "imag", "Kmag"], ["ymag", "ymag", "Kmag"], ["Jmag", "Jmag", "Kmag"]]

from_color = [[0.74, 0.74, 0.74], [0.62, 0.79, 0.88], [0.72, 0.78, 0.71]]
to_color = [[0.27, 0.27, 0.27], [0.0, 0.25, 0.53], [0.17, 0.36, 0.25]]

# isochrone_colors = ["#e7298a", "#66a61e"]
isochrone_colors = ["#7fc97f", "#e7298a"]
labels = ["Pleiades", "IC 4665"]

for k, filters in enumerate(CMD_combis[:]):
    for i, cluster in enumerate(case_study_names[:]):
        OC = star_cluster(cluster, N_df, catalog_mode=False)
        OC.create_CMD(CMD_params=filters, no_errors=True)

        setattr(OC, "weights", np.ones(len(OC.CMD[:, 0])))

        OC_density_x, OC_density_y, OC_kwargs = CMD_density_design(OC.CMD, to_RBG=to_color[i], from_RBG=from_color[i],
                                                                   cluster_obj=OC, density_plot=False)
        OC_kwargs["s"] = 20

        # 3. Do some initial HP tuning if necessary
        try:
            params = OC.SVR_read_from_file(HP_file_cs, False)
        except IndexError:
            print(OC.name)
            curve, isochrone = OC.curve_extraction(OC.PCA_XY, **kwargs)

        n_boot = 1000
        result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs,
                                               output_loc="data/Isochrones/Empirical/Case_studies/")

        axes[k].scatter(OC_density_x, OC_density_y, label=labels[i], **OC_kwargs)
        axes[k].set_xlabel((OC.CMD_specs["axes"][1]).replace("-", " - "))
        axes[k].set_ylabel(r"$\mathrm{}_{}$".format("{M}",
                                                    "\mathrm{" + OC.CMD_specs["axes"][0].split("mag")[0] + "}"),
                           labelpad=1)

        axes[k].plot(result_df["l_x"], result_df["l_y"], color="black", alpha=0.75, lw=1)  # label="5. perc", lw=1)
        axes[k].plot(result_df["u_x"], result_df["u_y"], color="black", alpha=0.75, label="5p/95p bounds", lw=1)
        axes[k].plot(result_df["m_x"], result_df["m_y"], color=isochrone_colors[i],
                     label="{} isochrone".format(labels[i]))

axes[0].set_xlim(-0.5, 3.5)
axes[0].set_ylim(19, 3)

axes[1].set_xlim(0, 2.5)
axes[1].set_ylim(19, 5)

axes[2].set_xlim(0, 4)
axes[2].set_ylim(18, 5)

axes[3].set_xlim(1, 7)
axes[3].set_ylim(18, 5)

axes[4].set_xlim(1, 4)
axes[4].set_ylim(15, 4)

axes[5].set_xlim(0.25, 1.75)
axes[5].set_ylim(13, 3)

handles, labels = axes[5].get_legend_handles_labels()
handles_new = [handles[i] for i in [3, 0, 5, 2, 1]]
labels_new = [labels[i] for i in [3, 0, 5, 2, 1]]
# ax[2].legend(handles_new, labels_new, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.5)


# plt.legend(handles_new, labels_new, loc="upper center", bbox_to_anchor=(-0.75, 2.43), ncol=3)
plt.legend(handles_new, labels_new, loc="upper center", bbox_to_anchor=(-0.825, 2.36), ncol=5)

fig.show()

if save_plot:
    fig.savefig(output_path + "Photometric_systems.pdf", dpi=600)
