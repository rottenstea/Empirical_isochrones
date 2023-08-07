import my_utility

import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from Empirical_iso_reader import merged_BPRP, merged_BPG, merged_GRP

from Classfile import *
from pre_processing import cluster_df_list

output_path = my_utility.set_output_path()
save_plot = True
# ----------------------------------------------------------------------------------------------------------------------

# Plot
# =======
# 2D Fig

# Poster figure (single panel)
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

palette_3 = ["#e7298a", "#66a61e", "#e6ab02"]
# palette_3 = ["#1b9e77", "#d95f02", "#7570b3"]

fig_poster, ax = plt.subplots(1, 1, figsize=(3.5, 5))

plt.subplots_adjust(left=0.2, bottom=0.12, top=0.99, right=0.975, wspace=0.0)
# sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
# sm.set_array([])
MG_comp = merged_BPRP[merged_BPRP["Cluster_id"].isin(["Meingast_1", "Melotte_22", "Blanco_1"])]

BPRP = sns.lineplot(data=MG_comp, x="m_x",
                    y="m_y", hue="Cluster_id", palette=palette_3, legend=True, sort=False,
                    lw=1, units="Cluster_id", ax=ax, estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                r"\mathrm{G}_{\mathrm{RP}}$",
                                                                         ylabel=r"$\mathrm{M}_{\mathrm{G}}$")

ax.set_xlim(-0.2, 4)
ax.set_ylim(15, -1)

# Get the current legend
legend = plt.gca().get_legend()

# Remove the title of the legend
legend.set_title("")

# c = fig_2D.colorbar(sm, location='right', fraction=0.1)
# cax = c.ax
# cax.text(0, 6.8, 'log age')
# c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
# cax = c.ax
# cax.text(6.65, 0.3, 'log age')
fig_poster.show()
if save_plot:
    fig_poster.savefig(output_path + "MG1.pdf", dpi=600)

# tryout

OC1 = star_cluster("Melotte_22", cluster_df_list[1], catalog_mode=True)
OC1.create_CMD(CMD_params=["Gmag", "BP-RP"])
OC1.kwargs_CMD["s"] = 20

OC2 = star_cluster("Blanco_1", cluster_df_list[1], catalog_mode=True)
OC2.create_CMD(CMD_params=["Gmag", "BP-RP"])
OC2.kwargs_CMD["s"] = 20

OC3 = star_cluster("Meingast_1", cluster_df_list[5], catalog_mode=False)
OC3.create_CMD(CMD_params=["Gmag", "BP-RP"])
OC3.kwargs_CMD["s"] = 20

fig_poster_2 = plt.figure(figsize=(7.24551, 6.5))

ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
ax2 = plt.subplot2grid((3, 3), (0, 2))
ax3 = plt.subplot2grid((3, 3), (1, 2), sharey=ax2)
ax4 = plt.subplot2grid((3, 3), (2, 2), sharey=ax2)

plt.subplots_adjust(left=0.06, bottom=0.07, top=0.99, right=0.98, hspace=0.1, wspace=0.15)

ax2.scatter(OC2.CMD[:, 0], OC2.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax2.scatter(OC3.CMD[:, 0], OC3.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax2.scatter(OC1.density_x, OC1.density_y, **OC1.kwargs_CMD, label="Pleiades")

ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax2.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_y"],
         color=palette_3[0])

ax2.set_xlim(-0.5, 4)
ax2.set_ylim(14.8, -1.5)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax2.text(-0.4, 14, "Pleiades")

ax3.scatter(OC1.CMD[:, 0], OC1.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax3.scatter(OC3.CMD[:, 0], OC3.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax3.scatter(OC2.density_x, OC2.density_y, **OC2.kwargs_CMD, label=OC2.name)

ax3.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax3.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax3.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_y"],
         color=palette_3[1])
ax3.set_xlim(-0.5, 4)
ax3.set_ylim(14.8, -1.5)
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax3.text(-0.4, 14, "Blanco 1")

ax4.scatter(OC1.CMD[:, 0], OC1.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax4.scatter(OC2.CMD[:, 0], OC2.CMD[:, 1], color="grey", marker=".", s=5, alpha=0.3)
ax4.scatter(OC3.density_x, OC3.density_y, **OC3.kwargs_CMD, label=OC3.name)

ax4.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax4.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_y"],
         color="black", lw=1, ls="solid", alpha=0.75)
ax4.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_y"],
         color=palette_3[2])
ax4.set_xlim(-0.5, 4)
ax4.set_ylim(14.8, -1.5)
ax4.set_yticklabels([])
ax4.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
ax4.text(-0.4, 14, "Meingast 1")

# uncertainty interval
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["l_y"],
         color=palette_3[0], lw=1, ls="--", alpha=0.75)
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["u_y"],
         color=palette_3[0], lw=1, ls="--", alpha=0.75)

ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["l_y"],
         color=palette_3[1], lw=1, ls="--", alpha=0.75)
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["u_y"],
         color=palette_3[1], lw=1, ls="--", alpha=0.75)

ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["l_y"],
         color=palette_3[2], lw=1, ls="--", alpha=0.75)
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["u_y"],
         color=palette_3[2], lw=1, ls="--", alpha=0.75)

# empirical isochrones
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC1.name]["m_y"],
         color=palette_3[0], label="Pleiades")
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC2.name]["m_y"],
         color=palette_3[1], label=OC2.name.replace("_", " "))
ax1.plot(MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_x"], MG_comp[MG_comp["Cluster_id"] == OC3.name]["m_y"],
         color=palette_3[2], label=OC3.name.replace("_", " "))
ax1.set_xlim(-0.5, 4)
ax1.set_ylim(14.8, -1.5)
ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
ax1.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$", labelpad=1)
ax1.legend(loc="upper right")

fig_poster_2.show()
if save_plot:
    fig_poster_2.savefig(output_path + "Age_brackets_MG1.pdf", dpi=600)

'''
# 3D Plot
# ============
merged_BPRP = merged_BPRP[(merged_BPRP["ref_age"] >= age_low) & (merged_BPRP["ref_age"] <= age_up)]
merged_BPG = merged_BPG[(merged_BPG["ref_age"] >= age_low) & (merged_BPG["ref_age"] <= age_up)]
merged_GRP= merged_GRP[(merged_GRP["ref_age"] >= age_low) & (merged_GRP["ref_age"] <= age_up)]


sorted_clusters = merged_GRP["Cluster_id"].drop_duplicates()
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
a = plt.get_cmap("winter", len(sorted_clusters))

rgbs = a(range(len(sorted_clusters) + 40))
col_hex = [colors.rgb2hex(c) for c in rgbs]
colo_hex = col_hex[:84]
# line_width = [1]* len(colo_hex)
# line_width.insert(8,2)
# line_width.insert(9,2)
#
# colo_hex.insert(8,"firebrick")
# colo_hex.insert(9,"darkorange")

BPRP_x = [merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_x'] for cluster in sorted_clusters]
BPRP_y = [merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_y'] for cluster in sorted_clusters]

BPG_x = [merged_BPG[merged_BPG["Cluster_id"] == cluster]['m_x'] for cluster in sorted_clusters]
BPG_y = [merged_BPG[merged_BPG["Cluster_id"] == cluster]['m_y'] for cluster in sorted_clusters]

RPG_x = [merged_GRP[merged_GRP["Cluster_id"] == cluster]['m_x'] for cluster in sorted_clusters]
RPG_y = [merged_GRP[merged_GRP["Cluster_id"] == cluster]['m_y'] for cluster in sorted_clusters]

# create initial plot
fig = go.Figure()
fig.update_xaxes(range=[-1, 4])
fig.update_yaxes(range=[15, -4])

fig.update_layout(width=680, height=780)
fig.update_layout(template='plotly_white')


cluster_labels = []
for i, cluster in enumerate(sorted_clusters):
    if "_" in cluster:
        c_label = "%s: %s Myr" % (cluster.replace("_", " "), round(10 ** (
            merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                "ref_age"].unique()[0]) / 1e6, 2))
        # c_label = r"$\text{%s}: %s \text{ Myr}$" % (cluster.replace("_", " "), round(10 ** (
        #   merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
        #       "ref_age"].unique()[0]) / 1e6, 2))

    # elif any([key in cluster for key in greek_dict.keys()]):
    #     for key in greek_dict.keys():
    #         if key in str(cluster):
    #             split = cluster.split(" ", maxsplit=2)
    #             greek = split[0].replace(key, greek_dict[key])
    #             c_label = r"$ %s \text{%s}: %s \text{ Myr}$" % (greek, " " + split[1], round(10 ** (
    #                 merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
    #                     "ref_age"].unique()[0]) / 1e6, 2))
    else:
        c_label = "%s: %s Myr" % (cluster, round(10 ** (
            merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                "ref_age"].unique()[0]) / 1e6, 2))
    cluster_labels.append(c_label)

    trace = go.Scatter(x=merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_x'],
                       y=merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_y'],
                       name=c_label,
                       visible=True,  # make the first line visible by default
                       line=dict(color=colo_hex[i], width = line_width[i]))
    fig.add_trace(trace)

# create button layout
updatemenus = [
    dict(
        type='buttons',
        x=0.75,
        y=-0.05,
        direction="right",
        showactive=True,
        buttons=[
            dict(
                label=r'BP-RP',
                method='update',
                args=[{'x': BPRP_x, 'y': BPRP_y, 'name': cluster_labels},
                      # ["{0}: {1} Myr".format(cluster, round(10 ** (
                      # merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                      #    "ref_age"].unique()[0]) / 1e6, 2)) for cluster in sorted_clusters]},
                      {"xaxis.range": [-2, 4]}]
            ),
            dict(
                label='BP-G',
                method='update',
                args=[{'x': BPG_x, 'y': BPG_y, 'name': cluster_labels},
                      # ["{0}: {1} Myr".format(cluster, round(10 ** ( merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                      #    "ref_age"].unique()[0]) / 1e6, 2)) for cluster in sorted_clusters]},
                      {"xaxis.ramge": [-1, 3]}]
            ),
            dict(
                label='G-RP',
                method='update',
                args=[{'x': RPG_x, 'y': RPG_y, 'name': cluster_labels},
                      # ["{0}: {1} Myr".format(cluster, round(10 ** ( merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                      #     "ref_age"].unique()[0]) / 1e6, 2)) for cluster in sorted_clusters]},
                      {"xaxis.ramge": [0, 3]}]
            )
        ]
    )
]

# update layout with button layout
fig.update_layout(updatemenus=updatemenus)

# Add slider
steps = []

for i, cluster in enumerate(sorted_clusters):
    df = merged_BPRP[merged_BPRP["Cluster_id"] == cluster]
    df = df.sort_values("m_y")

    visible_traces = [j <= i + 1 for j in range(len(sorted_clusters))]
    step = dict(
        method="update",
        args=[{"visible": visible_traces}],
        label=df["ref_age"].unique()[0])
    steps.append(step)

sliders = [dict(
    active=0,
    y=-0.02,
    currentvalue={"prefix": "log age: "},
    pad={"t": 50},
    steps=steps,
)]

# update layout with slider
fig.update_layout(
    sliders=sliders,
    # xaxis_title="BP-RP",
    yaxis_title="abs Gmag",

    autosize=True,
    margin=dict(l=50, r=50, t=50, b=50),
    title={
        'text': 'Empirical isochrones',
        'x': 0.5,
        'xanchor': 'center'
    }
)

# plot the fig
plot(fig, filename=output_path + 'ScoCen_analysis.html')
# Write the figure to an HTML file
#if save_plot:
    #fig.write_html(output_path + 'Meingast1_analysis.html', include_mathjax='cdn')
'''
