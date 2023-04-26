from pre_processing import cluster_df_list, cluster_name_list
from Classfile import *
import my_utility

import numpy as np

import plotly.express as px
import seaborn as sns
from plotly.offline import plot
import matplotlib.pyplot as plt

output_path = my_utility.set_output_path()

savefig = False

CI_df, CII_df, CIII_df, AOI_df, AOII_df, AOIII_df = cluster_df_list
CI_names, CII_names, CIII_names, AOI_names, AOII_names, AOIII_names = cluster_name_list

# for XYZ plotting
xyz_list = []

for df in cluster_df_list:
    xyz = df.drop_duplicates(subset="Cluster_id")
    xyz_list.append(xyz)
    # print(xyz.shape)

# 1. Fuse
Archive_clusters = np.concatenate([cluster_name_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
Archive_df = pd.concat(xyz_list, axis=0)

min_age = np.min(Archive_df["ref_age"])
max_age = np.max(Archive_df["ref_age"])

test = sns.color_palette("crest", n_colors=9).as_hex()

# 3D plot
fig = px.scatter_3d(Archive_df, x='x', y='y', z='z',
                    color='ref_age', size='Nstars', size_max=50,
                    symbol='catalog', opacity=0.7, color_continuous_scale=test,
                    symbol_sequence=("circle", "circle-open", "square", "diamond"),
                    hover_data=[Archive_df["Cluster_id"]])

# tight layout
fig.update_xaxes(range=[-500, 500])
fig.update_yaxes(range=[-500, 500])
fig.update_yaxes(range=[-500, 500])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), template='plotly_white',
                  coloraxis={"colorbar": {"x": -0.2, "len": 0.5, "y": 0.8}},
                  xaxis_title="X / pc", yaxis_title="Y / pc",
                  autosize=True,
                  title={'text': 'Cluster position and ages',
                         'x': 0.5, 'xanchor': 'center'})

plot(fig, filename=output_path + 'XYZ_express.html')

# fig.write_html(output_path+"XYZ_express.html")

# 2D plot

fig_2D, ax = plt.subplots(1, 3, layout="constrained")

a = plt.get_cmap("YlGnBu_r")

sns.set_style("darkgrid")
norm = plt.Normalize(Archive_df['ref_age'].min(), Archive_df['ref_age'].max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

xy = sns.scatterplot(
    data=Archive_df, x="x", y="y", size="Nstars", hue="ref_age", palette=a, norm=norm,
    sizes=(20, 200), legend=False, ax=ax[0],  # style = "catalog",
).set(xlabel="X (pc)", ylabel="Y (pc)")
ax[0].set_xlim(-525, 525)
ax[0].set_ylim(-525, 525)
ax[0].set_aspect("equal")

yz = sns.scatterplot(
    data=Archive_df, x="y", y="z", size="Nstars", hue="ref_age", ax=ax[1], norm=norm, palette=a,
    # style = "catalog",
    sizes=(20, 200), legend=False).set(xlabel="Y (pc)", ylabel="Z (pc)")
ax[1].set_xlim(-525, 525)
ax[1].set_ylim(-525, 525)
ax[1].set_aspect("equal")
ax[1].xaxis.grid(True, "minor", linewidth=.25)
ax[1].yaxis.grid(True, "minor", linewidth=.25)
ax[1].yaxis.set_ticklabels([])

xz = sns.scatterplot(
    data=Archive_df, x="x", y="z", size="Nstars", hue="ref_age", legend=True,
    sizes=(20, 200), ax=ax[2], norm=norm, palette=a,  # style = "catalog",
).set(xlabel="X (pc)", ylabel="Z (pc)")
# xz.ax.scatter(0, 0, marker="*", color="black", s=50, label="barycenter of the sun")
ax[2].set_xlim(-525, 525)
ax[2].set_ylim(-525, 525)
ax[2].xaxis.grid(True, "minor", linewidth=.25)
ax[2].yaxis.grid(True, "minor", linewidth=.25)
ax[2].set_aspect("equal")
ax[2].yaxis.set_ticklabels([])

# Remove the legend and add a colorbar
ax[2].get_legend().remove()
c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
cax = c.ax
cax.text(6.6, 0.3, 'log age')

handles, labels = ax[2].get_legend_handles_labels()
entries_to_skip = 7
handles_new = handles[entries_to_skip:-1]
labels_new = labels[entries_to_skip:-1]
labs = labels_new[:1] + [f'{int(lab)}' for lab in labels_new[1:]]
ax[2].legend(handles_new, labs, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)

fig_2D.set_figheight(4)
fig_2D.set_figwidth(10)
sns.set(font_scale=1.2)

plt.show()

if savefig:
    fig_2D.savefig(output_path + "XYZ_plot_CGvalues_all.pdf", dpi=600)

# Extinction
def extinction_plot(dataframe, extinction_vec, arrow_pos, params, title_ax_labels, extinction_mag: str = "G",
                    rotate: int = -44, figure_specs=None):
    if figure_specs is None:
        figure_specs = [5, 6]

    CMD_parameters = [params[1], params[2]]
    parallax = params[0]

    distance = 1000 / dataframe[parallax].to_numpy()
    mag, color_ax = dataframe[CMD_parameters[0]].to_numpy(), dataframe[CMD_parameters[1]].to_numpy()
    abs_mag = (mag - 5 * np.log10(distance) + 5)
    arr = np.stack([color_ax, abs_mag], axis=1)
    cleaned_arr = arr[~np.isnan(arr).any(axis=1), :]
    sorted_arr = cleaned_arr[cleaned_arr[:, 1].argsort()]
    f = CMD_density_design(sorted_arr, fig_specs=figure_specs,
                           title_axes_specs=title_ax_labels)

    plt.annotate("", xy=(arrow_pos[0] + extinction_vec[0], arrow_pos[1] + extinction_vec[1]),
                 xytext=(arrow_pos[0], arrow_pos[1]),
                 arrowprops=dict(arrowstyle="->", color="k"))
    plt.text(arrow_pos[0] + 0.1, arrow_pos[1] + 0.1, "A$_{0}$=1".format(extinction_mag), size=8,
             rotation=rotate)

    return f


CMD_params = ["plx_DR2", "Gmag_DR2", "BP-RP_DR2"]
extinction_DR2 = [0.5290999999999997, 1]

title_ax_CI = ["Catalog I sources / DR 2 (Cantat-Gaudin+2020)", "BP-RP", "abs Gmag"]
CI_extinction_fig = extinction_plot(dataframe=CI_df, extinction_vec=extinction_DR2,
                                    arrow_pos=[3.5, 7], params=CMD_params, title_ax_labels=title_ax_CI)

title_ax_CIII = ["Sco-Cen clusters / DR2", "BP-RP", "abs Gmag"]
CIII_extinction_fig = extinction_plot(dataframe=CIII_df, extinction_vec=extinction_DR2,
                                      arrow_pos=[4, 8], params=CMD_params, title_ax_labels=title_ax_CIII)

CI_extinction_fig.show()
CIII_extinction_fig.show()

if savefig:
    CI_extinction_fig.savefig(output_path + "Extinction_DR2_CI.pdf", dpi=600, bbox_inches ="tight")
    CIII_extinction_fig.savefig(output_path + "Extinction_DR2_CIII.pdf", dpi=600, bbox_inches="tight")

