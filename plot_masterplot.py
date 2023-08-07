import my_utility

import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from Empirical_iso_reader import merged_BPRP, merged_BPG, merged_GRP

output_path = my_utility.set_output_path()
save_plot = True
# ----------------------------------------------------------------------------------------------------------------------

merged_BPRP = merged_BPRP[merged_BPRP.Cluster_id != "Meingast_1_ESSII"]
merged_BPG = merged_BPG[merged_BPG.Cluster_id != "Meingast_1_ESSII"]
merged_GRP = merged_GRP[merged_GRP.Cluster_id != "Meingast_1_ESSII"]

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

age_low = 6
age_up = 11

fig_2D, ax = plt.subplots(1, 3, figsize=(7.24551, 4.4), sharey="row")

plt.subplots_adjust(left=0.08, bottom=0.0, top=0.99, right=0.99, wspace=0.1)

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
c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35, pad=0.12)
cax = c.ax
cax.tick_params(labelsize=10)
cax.text(6.65, 0.3, 'log age')

fig_2D.show()
if save_plot:
    fig_2D.savefig(output_path + "Summary_plot_C2.pdf", dpi=600)
'''

# Poster figure (single panel)
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 12

a = plt.get_cmap("YlGnBu_r")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

age_low = 6
age_up = 11

fig_poster, ax = plt.subplots(1, 1, figsize=(3.5, 4.4))

plt.subplots_adjust(left=0.16,bottom=0.11, top = 0.98, right=0.97, wspace=0.0)

BPRP = sns.lineplot(data=merged_BPRP, x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax, estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$",
                                                                            ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax.set_xlim(-0.5, 4.1)
ax.set_ylim(14, -3.5)

c = fig_poster.colorbar(sm, ax=ax, location='right', fraction=0.15, aspect=20, pad=0.05)
cax = c.ax
plt.text(5.2, 18, 'log age')

fig_poster.show()
if save_plot:
    fig_poster.savefig(output_path + "C2_summary_plot.pdf", dpi=600)


# ============
# 3D Fig

sorted_clusters = merged_GRP["Cluster_id"].drop_duplicates()
age_range = merged_BPRP.drop_duplicates(subset="Cluster_id")
#norm = plt.Normalize(age_range.min(), age_range.max())
#a = plt.get_cmap("YlGnBu_r", len(sorted_clusters))
cm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm).to_rgba(age_range["ref_age"], alpha=None, bytes=False, norm=True)

#rgbs = cm(range(len(sorted_clusters)))
col_hex = [colors.rgb2hex(c) for c in cm]
colo_hex = col_hex[:]

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

#fig.update_layout(width=680, height=780)
fig.update_layout(width=750, height=850)

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

    unified_label= f"{c_label:_<25}"
    #unified_label = c_label.ljust(25)
    cluster_labels.append(unified_label)

    trace = go.Scatter(x=merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_x'],
                       y=merged_BPRP[merged_BPRP["Cluster_id"] == cluster]['m_y'],
                       name=unified_label,
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
    yaxis_title="absolute G magnitude",

    autosize=False,
    margin=dict(l=50, r=50, t=50, b=50),
    title={
        'text': 'Empirical isochrones',
        'x': 0.5,
        'xanchor': 'center'
    }
)


fig.update_layout(
    plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
    xaxis=dict(gridcolor='white'),      # Set the x-axis grid line color to white
    yaxis=dict(gridcolor='white'),      # Set the y-axis grid line color to white
)

# plot the fig
plot(fig, filename=output_path + 'Empirical_isochrone_archive.html')
# Write the figure to an HTML file
if save_plot:
    fig.write_html(output_path + 'Empirical_isochrone_archive.html')
'''
