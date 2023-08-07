import my_utility

import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from Empirical_iso_reader import build_empirical_df, reference_ages, empirical_iso_path

output_path = my_utility.set_output_path()
save_plot = True

threeD = False
# ----------------------------------------------------------------------------------------------------------------------

general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                      name_split="_G")

# BP - RP
# ========

merged_BPRP = build_empirical_df(filename_key="G_BPRP_nboot_1000_cat_3", **general_kwargs)

# BP - G
# ========

merged_BPG = build_empirical_df(filename_key="G_BPG_nboot_1000_cat_3", **general_kwargs)

# G - RP
# ========

merged_GRP = build_empirical_df(filename_key="G_GRP_nboot_1000_cat_3", **general_kwargs)

# Plot
# =======
# 2D Fig

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

a = plt.get_cmap("YlGnBu_r")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])
'''
fig_2D, ax = plt.subplots(1, 3, figsize=(6, 4), sharey="row")

BPRP = sns.lineplot(data=merged_BPRP, x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$",
                                                                            ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[0].set_xlim(-1., 4.5)
ax[0].set_ylim(13, -3.)

# ax[0].set_aspect("equal")

BPG = sns.lineplot(data=merged_BPG, x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                  r"\mathrm{G}$")
ax[1].set_xlim(-.5, 3.)
ax[1].set_ylim(13, -3)

RPG = sns.lineplot(data=merged_GRP, x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[2], estimator=None).set(xlabel=r"$\mathrm{G}- \mathrm{G}_{"
                                                                                  r"\mathrm{RP}}$")
ax[2].set_xlim(-0.3, 1.8)
ax[2].set_ylim(13, -3)

plt.subplots_adjust(left=0.09, right=0.98, bottom=0.02, top=0.98, wspace=0.1)

c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
cax = c.ax
plt.text(-5.6, 16.9, 'log age')

# fig_2D.show()
if save_plot:
    fig_2D.savefig(output_path + "ScoCen_analysis_2D_logage.pdf", dpi=600)
'''
# 1 Panel 2D plot
# ================

fig_1Panel = plt.figure(figsize=(3.54399, 4.7))

ax = plt.subplot2grid((1, 1), (0, 0))
# ax2 = plt.subplot2grid((1, 2), (0, 1))

plt.subplots_adjust(left=0.13, right=0.93, bottom=0.09, top=0.99)

BPRP = sns.lineplot(data=merged_BPRP, x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax, estimator=None)  # .set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
#           r"\mathrm{G}_{\mathrm{RP}}$",
#    ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax.set_xlim(-1., 4.5)
ax.set_ylim(13.5, -3.)
# ax.set_title("Sco-Cen isochrones")
ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$", labelpad=2)

# zoom = sns.lineplot(data=merged_BPRP, x="m_x",
#                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
#                  lw=1, units="Cluster_id", ax=ax2, estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
#                                                                              r"\mathrm{G}_{\mathrm{RP}}$", ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
# ax2.set_xlim(.5, 3.5)
# ax2.set_ylim(12, 3)


c = fig_1Panel.colorbar(sm, ax=ax, location='right', fraction=0.1, aspect=20)
c.ax.tick_params(labelsize=10)
plt.text(4.65, 14.3, 'log age')

fig_1Panel.show()
if save_plot:
    fig_1Panel.savefig(output_path + "ScoCen_analysis.pdf", dpi=600)

# Zoom in
# ========
'''
fig_zoom, ax = plt.subplots(2, 1, figsize=(3, 5),gridspec_kw={'height_ratios': [3, 1]})

BPRP = sns.lineplot(data=merged_BPRP, x="m_x",
                    y="m_y", hue="ref_age", palette="winter", hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel="", ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[0].set_xlim(-1., 4.5)
ax[0].set_ylim(13.5, -3.)
#

zoom = sns.lineplot(data=merged_BPRP, x="m_x",
                   y="m_y", hue="ref_age", palette="winter", hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$", ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[1].set_xlim(.5, 3.5)
ax[1].set_ylim(12, 3)

plt.subplots_adjust(left=0.16, right=0.92, bottom=0.09, top=0.98, hspace=0.13)
#
c = fig_2D.colorbar(sm, ax=[ax[0], ax[1]], location='right', fraction=0.1, aspect=35)
# cax = c.ax
ax[1].text(3.5, 13.2, 'log age')

fig_zoom.show()
if save_plot:
    fig_zoom.savefig(output_path + "ScoCen_analysis_zoom_logage.pdf", dpi=600)
'''

if threeD:

    # 3D Plot
    # ============
    sorted_clusters = merged_GRP["Cluster_id"].drop_duplicates()
    age_range = merged_BPRP.drop_duplicates(subset="Cluster_id")
    # norm = plt.Normalize(age_range.min(), age_range.max())
    # a = plt.get_cmap("YlGnBu_r", len(sorted_clusters))
    cm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm).to_rgba(age_range["ref_age"], alpha=None, bytes=False,
                                                                   norm=True)

    # rgbs = cm(range(len(sorted_clusters)))
    col_hex = [colors.rgb2hex(c) for c in cm]
    colo_hex = col_hex[:]
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

    fig.update_layout(width=750, height=850)
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
                           line=dict(color=colo_hex[i], width=2))
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
            'text': 'A closer look into the Sco-Cen ages',
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    fig.update_layout(
        plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
        xaxis=dict(gridcolor='white'),  # Set the x-axis grid line color to white
        yaxis=dict(gridcolor='white'),  # Set the y-axis grid line color to white
    )

    # plot the fig
    plot(fig, filename=output_path + 'ScoCen_analysis.html')
    # Write the figure to an HTML file
    # if save_plot:
    # fig.write_html(output_path + 'Meingast1_analysis.html', include_mathjax='cdn')
