import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objs as go
from plotly.offline import plot
import seaborn as sns

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Empirical_iso_reader import build_empirical_df, reference_ages, empirical_iso_path

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
# import the isochrones from the results file
general_kwargs = dict(csv_folder=empirical_iso_path, age_file=reference_ages, col_names=["ref_age", "m_y"],
                      name_split="_G")

merged_BPRP = build_empirical_df(filename_key="G_BPRP_nboot_1000_cat_3", **general_kwargs)
merged_BPG = build_empirical_df(filename_key="G_BPG_nboot_1000_cat_3", **general_kwargs)
merged_GRP = build_empirical_df(filename_key="G_GRP_nboot_1000_cat_3", **general_kwargs)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# 2D plot

# define the norm of the colorbar using the age info
a = plt.get_cmap("YlGnBu_r")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

Ladder_fig = plt.figure(figsize=(3.54399, 4.7))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.subplots_adjust(left=0.13, right=0.93, bottom=0.09, top=0.99)

BPRP = sns.lineplot(data=merged_BPRP, x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax, estimator=None)
ax.set_xlim(-1., 4.5)
ax.set_ylim(13.5, -3.)
ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)")
ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$ (mag)", labelpad=2)

# set colorbar
c = Ladder_fig.colorbar(sm, ax=ax, location='right', fraction=0.1, aspect=20)
c.ax.tick_params(labelsize=10)
plt.text(4.65, 14.3, 'log age')

Ladder_fig.show()
if save_plot:
    Ladder_fig.savefig(output_path + "ScoCen_analysis.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Interactive version
if interactive:

    # get namelist + colorscheme of sorted clusters
    sorted_clusters = merged_GRP["Cluster_id"].drop_duplicates()
    age_range = merged_BPRP.drop_duplicates(subset="Cluster_id")
    cm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm).to_rgba(age_range["ref_age"], alpha=None, bytes=False,
                                                                   norm=True)
    col_hex = [colors.rgb2hex(c) for c in cm]
    colo_hex = col_hex[:]

    # a bit convoluted but for each cluster collect the XY data from the result dataframe in all passband combinations
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

    # create custom cluster labels for the legend that also show the ages
    cluster_labels = []
    for i, cluster in enumerate(sorted_clusters):
        if "_" in cluster:
            c_label = "%s: %s Myr" % (cluster.replace("_", " "), round(10 ** (
                merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                    "ref_age"].unique()[0]) / 1e6, 2))

        else:
            c_label = "%s: %s Myr" % (cluster, round(10 ** (
                merged_BPRP[merged_BPRP["Cluster_id"] == cluster][
                    "ref_age"].unique()[0]) / 1e6, 2))
        cluster_labels.append(c_label)

        # add all traces to the plot in the BP-RP (appears when first opening the plot)
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
                    args=[{'x': BPRP_x, 'y': BPRP_y, 'name': cluster_labels}, {"xaxis.range": [-2, 4]}]),
                dict(
                    label='BP-G',
                    method='update',
                    args=[{'x': BPG_x, 'y': BPG_y, 'name': cluster_labels}, {"xaxis.ramge": [-1, 3]}]),
                dict(
                    label='G-RP',
                    method='update',
                    args=[{'x': RPG_x, 'y': RPG_y, 'name': cluster_labels}, {"xaxis.ramge": [0, 3]}])
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
        yaxis_title="abs Gmag",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
        title={
            'text': 'A closer look into the Sco-Cen ages',
            'x': 0.5,
            'xanchor': 'center'}
    )

    # Layout in white and grey
    fig.update_layout(
        plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
        xaxis=dict(gridcolor='white'),  # Set the x-axis grid line color to white
        yaxis=dict(gridcolor='white'),  # Set the y-axis grid line color to white
    )

    # plot the PCA_matrix
    plot(fig, filename=output_path + 'ScoCen_analysis.html')
# ----------------------------------------------------------------------------------------------------------------------