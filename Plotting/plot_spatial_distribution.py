import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

from My_tools import my_utility
from Extraction.Classfile import *
from Extraction.pre_processing import cluster_df_list, cluster_name_list

# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

save_plot = False
interactive = True
# ----------------------------------------------------------------------------------------------------------------------
# DATA CRUNCHING
# ----------------------------------------------------------------------------------------------------------------------
# for XYZ plotting
xyz_list = []
for df in cluster_df_list:
    xyz = df.drop_duplicates(subset="Cluster_id")
    xyz_list.append(xyz)

# Fuse
Archive_clusters = np.concatenate([cluster_name_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
Archive_df = pd.concat(xyz_list, axis=0)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# 2D plot
fig_2D, ax = plt.subplots(1, 3, figsize=(7.24551, 2.6))
plt.subplots_adjust(top=0.99, left=0.085, right=0.87, bottom=0.055, wspace=0.25)
a = plt.get_cmap("YlGnBu_r")

# define the norm of the colorbar using the age info
norm = plt.Normalize(Archive_df['ref_age'].min(), Archive_df['ref_age'].max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

# XY subplot
xy = sns.scatterplot(
    data=Archive_df, x="x", y="y", size="Nstars", hue="ref_age", palette=a, norm=norm,
    sizes=(20, 200), legend=False, ax=ax[0],  # style = "catalog",
).set(xlabel="X (pc)", ylabel="Y (pc)")
ax[0].set_xlim(-525, 525)
ax[0].set_ylim(-525, 525)
ax[0].set_aspect("equal")

# YZ subplot
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

# XZ subplot
xz = sns.scatterplot(
    data=Archive_df, x="x", y="z", size="Nstars", hue="ref_age", legend=True,
    sizes=(20, 200), ax=ax[2], norm=norm, palette=a,  # style = "catalog",
).set(xlabel="X (pc)", ylabel="Z (pc)")
ax[2].set_xlim(-525, 525)
ax[2].set_ylim(-525, 525)
ax[2].xaxis.grid(True, "minor", linewidth=.25)
ax[2].yaxis.grid(True, "minor", linewidth=.25)
ax[2].set_aspect("equal")
ax[2].yaxis.set_ticklabels([])

# Remove the legend and add a colorbar
ax[2].get_legend().remove()
c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35, pad=0.2)
cax = c.ax
cax.tick_params(labelsize=10)
cax.text(6.6, 0.2, 'log age')

# edit the legend entries so that only the marker size ond not the color info is added
handles, labels = ax[2].get_legend_handles_labels()
entries_to_skip = 7
handles_new = handles[entries_to_skip:-1]
labels_new = labels[entries_to_skip:-1]
labs = labels_new[:1] + [f'{int(lab)}' for lab in labels_new[1:]]
ax[2].legend(handles_new, labs, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.5)

plt.show()
if save_plot:
    fig_2D.savefig(output_path + "XYZ_plot_CGvalues.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# 3D plot
if interactive:
    min_age = np.min(Archive_df["ref_age"])
    max_age = np.max(Archive_df["ref_age"])
    palette = sns.color_palette("crest", n_colors=9).as_hex()

    # 3D plot
    fig = px.scatter_3d(Archive_df, x='x', y='y', z='z',
                        color='ref_age', size='Nstars', size_max=50,
                        symbol='catalog', opacity=0.7, color_continuous_scale=palette,
                        symbol_sequence=("circle", "circle-open", "square", "diamond"),
                        hover_data=[Archive_df["Cluster_id"]])

    # tight layout
    fig.update_xaxes(range=[-500, 500])
    fig.update_yaxes(range=[-500, 500])
    fig.update_yaxes(range=[-500, 500])

    # labels and layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), template='plotly_white',
                      coloraxis={"colorbar": {"x": -0.2, "len": 0.5, "y": 0.8}},
                      xaxis_title="X / pc", yaxis_title="Y / pc",
                      autosize=True,
                      title={'text': 'Cluster position and ages',
                             'x': 0.5, 'xanchor': 'center'})

    plot(fig, filename=output_path + 'XYZ_express.html')
# ----------------------------------------------------------------------------------------------------------------------
