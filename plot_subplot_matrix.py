import my_utility

import matplotlib.pyplot as plt
import seaborn as sns

from Classfile import *
from pre_processing import cluster_df_list, cluster_name_list
from Empirical_iso_reader import merged_BPRP

output_path = my_utility.set_output_path()

df_BPRP = merged_BPRP

Archive_clusters = np.concatenate([cluster_name_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
Archive_df = pd.concat([cluster_df_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
CII_df = pd.concat([cluster_df_list[i] for i in [1]], axis=0)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
kwargs = dict(grid=None, HP_file=None, catalog_mode=True)
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10
save_plot = True

# ----------------------------------------------------------------------------------------------------------------------
Fig, axes = plt.subplots(figsize=(7.24551, 8), nrows=5, ncols=8, sharex="col", sharey="row")

plt.subplots_adjust(left=0.057, right=0.995, top=0.97, bottom=0.053, wspace=0.1, hspace=0.23)

OCs = []

Archive_df.sort_values(by=["ref_age", "Cluster_id"], inplace=True)
sorted_names = Archive_df["Cluster_id"].unique().tolist()

# for n, cluster in enumerate(Archive_clusters[:]):
for n, cluster in enumerate(sorted_names):
    if not cluster in ["Blanco_1", "IC_2391", "IC_2602", "Melotte_20", "Melotte_22",
                       "NGC_2451A", "NGC_2516", "NGC_2547", "NGC_7092", "Platais_9"]:
        OC = star_cluster(cluster, Archive_df)
    else:
        OC = star_cluster(cluster, CII_df)

    # 2. Create the CMD that should be used for the isochrone extraction
    OC.create_CMD(CMD_params=["Gmag", "BPmag", "RPmag"])
    OCs.append(OC)

y_lims, x_lims = [], []

for i, ax in enumerate(axes.flat[:]):
    OC = OCs[i]
    kr = 0
    OC.kwargs_CMD["s"] = 20

    cs = ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label=OC.name.replace("_", " "))
    df = df_BPRP[df_BPRP["Cluster_id"] == OC.name]
    ax.plot(df["l_x"], df["l_y"], color="black", label="5. perc", alpha=0.75, lw=1)
    ax.plot(df["u_x"], df["u_y"], color="black", label="95. perc", alpha=0.75, lw=1)
    ax.fill_between(df["m_x"], df["l_y"], df["m_y"], color="grey", alpha=0.5, )
    ax.fill_between(df["m_x"], df["u_y"], df["l_y"], color="grey", alpha=0.5)
    ax.plot(df["m_x"], df["m_y"], color="red", lw=1)

    ax.set_title(OC.name.replace("_", " "))
    title = ax.title.get_text()
    if len(title) > 11:
        ax.title.set_fontsize(10)
for i in range(5):
    row = axes[i, :]
    y_lims, x_lims = [], []
    for axi in row:
        y_lims.extend(list(axi.get_ylim()))
        x_lims.extend(list(axi.get_xlim()))
    for axi in row:
        axi.set_ylim(max(y_lims), min(y_lims))
        axi.set_xlim(min(x_lims), max(x_lims))
        if i == 2:
            row[0].set_ylabel(r"M$_{\mathrm{G}}$", labelpad=1)
        if i == 4:
            row[3].set_xlabel("G$_{\mathrm{BP}}$ - G$_{\mathrm{RP}}$", x=1)

# Fig.show()
if save_plot:
    Fig.savefig(output_path + "Subplot_matrix_1-40.pdf", dpi=600)

Fig2, axes2 = plt.subplots(figsize=(7.24551, 9.2), nrows=6, ncols=8, sharex="col", sharey="row")

plt.subplots_adjust(left=0.057, right=0.995, top=0.97, bottom=0.048, wspace=0.1, hspace=0.23)

for i, ax in enumerate(axes2.flat[:]):
    try:

        OC = OCs[40 + i]
        kr = 0
        OC.kwargs_CMD["s"] = 20
        cs = ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label=OC.name.replace("_", " "))
        df = df_BPRP[df_BPRP["Cluster_id"] == OC.name]
        ax.plot(df["l_x"], df["l_y"], color="black", label="5. perc", alpha=0.75, lw=1)
        ax.plot(df["u_x"], df["u_y"], color="black", label="95. perc", alpha=0.75, lw=1)
        ax.fill_between(df["m_x"], df["l_y"], df["m_y"], color="grey", alpha=0.5, )
        ax.fill_between(df["m_x"], df["u_y"], df["l_y"], color="grey", alpha=0.5)
        ax.plot(df["m_x"], df["m_y"], color="red", lw=1)

        ax.set_title(OC.name.replace("_", " "))
        title = ax.title.get_text()
        if len(title) > 11:
            ax.title.set_fontsize(10)

    except IndexError:
        Fig2.delaxes(axes2.flat[i])

for i in range(6):
    row = axes2[i, :]
    y_lims, x_lims = [], []
    for ax in row:
        y_lims.extend(list(ax.get_ylim()))
        x_lims.extend(list(ax.get_xlim()))
    for ax in row:
        ax.set_ylim(max(y_lims), min(y_lims))
        ax.set_xlim(min(x_lims), max(x_lims))
        if i == 2:
            row[0].set_ylabel(r"M$_{\mathrm{G}}$", labelpad=1, y=-0.25)
        if i == 5:
            row[1].set_xlabel("G$_{\mathrm{BP}}$ - G$_{\mathrm{RP}}$")

Fig2.show()
if save_plot:
    Fig2.savefig(output_path + "Subplot_matrix_41-83.pdf", dpi=600)
