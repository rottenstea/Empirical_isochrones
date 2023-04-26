import my_utility

import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from Empirical_iso_reader import merged_BPRP, merged_BPG, merged_GRP

output_path = my_utility.set_output_path()
save_plot = False
# ----------------------------------------------------------------------------------------------------------------------

# Plot
# =======
# 2D Fig

a = plt.get_cmap("YlGnBu_r")
sns.set_style("darkgrid")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

age_low = 6
age_up = 10

fig_2D, ax = plt.subplots(1, 3, figsize=(8, 5), layout="constrained", sharey="row")

BPRP = sns.lineplot(data=merged_BPRP[(merged_BPRP["ref_age"] >= age_low) & (merged_BPRP["ref_age"] <= age_up)], x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$",
                                                                            ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[0].set_xlim(-1.9, 5)
ax[0].set_ylim(17, -3.5)

# ax[0].set_aspect("equal")

BPG = sns.lineplot(data=merged_BPG[(merged_BPG["ref_age"] >= age_low) & (merged_BPG["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                  r"\mathrm{G}$")
ax[1].set_xlim(-1.1, 3.2)
ax[1].set_ylim(17, -4.1)

RPG = sns.lineplot(data=merged_GRP[(merged_GRP["ref_age"] >= age_low) & (merged_GRP["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[2], estimator=None).set(xlabel=r"$\mathrm{G}- \mathrm{G}_{"
                                                                                  r"\mathrm{RP}}$")
ax[2].set_xlim(-0.5, 1.8)
ax[2].set_ylim(17, -4.2)

# c = fig_2D.colorbar(sm, location='right', fraction=0.1)
# cax = c.ax
# cax.text(0, 6.8, 'log age')
c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
cax = c.ax
cax.text(6.65, 0.3, 'log age')

fig_2D.show()
if save_plot:
    fig_2D.savefig(output_path + "Summary_plot_2D_logage.pdf", dpi=600)

# ============
# 3D Fig

sorted_clusters = merged_GRP["Cluster_id"].drop_duplicates()
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
a = plt.get_cmap("YlGnBu_r", len(sorted_clusters))

rgbs = a(range(len(sorted_clusters) + 40))
col_hex = [colors.rgb2hex(c) for c in rgbs]
colo_hex = col_hex[:84]

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

greek_dict = {"rho": r"\rho", "phi": r"\varphi", "beta": r"\beta",
              "delta": r"\delta", "eps": r"\varepsilon", "nu": r"\nu",
              "eta": r"\eta", "sig": r"\sigma", "sigma": r"\sigma"}

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
                       line=dict(color=colo_hex[i]))
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
plot(fig, filename=output_path + 'Empirical_isochrones_normal.html')
# Write the figure to an HTML file
if save_plot:
    fig.write_html(output_path + 'Empirical_isochrones_normal.html', include_mathjax='cdn')
