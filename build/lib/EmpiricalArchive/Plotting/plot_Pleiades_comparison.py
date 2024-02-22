import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *

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
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Load theoretical isochrones
Bossini = pd.read_csv(
    "//data/Isochrones/PARSEC_isochrones/Pleiades_plot_Dr2/Bossini.csv")
Bossini["BP-RP"] = Bossini["G_BPmag"] - Bossini["G_RPmag"]

Dias = pd.read_csv("//data/Isochrones/PARSEC_isochrones/Pleiades_plot_Dr2/Dias.csv")
Dias["BP-RP"] = Dias["G_BPftmag"] - Dias["G_RPmag"]
Dias["BP-RP2"] = Dias["G_BPbrmag"] - Dias["G_RPmag"]

CG = pd.read_csv(
    "//data/Isochrones/PARSEC_isochrones/Pleiades_plot_Dr2"
    "/CG_Evans_Av_0p18_age_7p87_Z_0p0152.csv")
CG["BP-RP"] = CG["G_BPmag"] - CG["G_RPmag"]

# Define Pleiades cluster object + CMD
Pleiades_df = pd.read_csv("//data/Cluster_data/Melotte_22_CG_DR2.csv")
Pleiades_df["Cluster_id"] = "Melotte_22"
OC = star_cluster("Melotte_22", Pleiades_df, catalog_mode=False)
OC.create_CMD(CMD_params=["Gmag", "BP-RP"], no_errors=True)

# change markersize for paper
OC.kwargs_CMD["s"] = 20
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(3.54399, 3.3))  # without CG

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

plt.subplots_adjust(bottom=0.115, top=0.865, right=0.985, hspace=0.23, wspace=0.23, left=0.122)  # without CG

axes = [ax1, ax2, ax3]

# cut-off values for the model isochrones
start = 2
stop = 200

for ax in axes:
    ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="Pleiades")
    ax.plot(Dias["BP-RP"][start:stop], Dias["Gmag"][start:stop], color="#e7298a", label="131 Myr")
    ax.plot(Bossini["BP-RP"][start:stop], Bossini["Gmag"][start:stop], color="#7fc97f", label="86 Myr")

lim = ax1.get_ylim()
ax1.set_ylim(14.2, -5)
ax1.set_xlim(-0.5, 5)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2.set_ylim(4, -5)
ax3.set_ylim(13, 7)

ax2.set_xlim(-.5, 2)
ax3.set_xlim(1, 4.5)

ax1.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1)
ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=1, x=1.)
ax1.legend(loc="upper center", bbox_to_anchor=(1, 1.2), ncol=3)  # without CG
plt.show()

if save_plot:
    fig.savefig(output_path + "Isochrone_comparison_Pleiades.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Poster plot
fig_poster, ax = plt.subplots(1, 3, figsize=(7.24551, 4.), sharey="row")
plt.subplots_adjust(left=0.08, bottom=0.13, top=0.92, right=0.975, wspace=0.15)

start = 2
stop = 200

# Subplot 1
ax[0].scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="data")
ax[0].plot(Dias["BP-RP"][start:stop], Dias["Gmag"][start:stop], color="#e7298a", label="131 Myr")
ax[0].text(-.4, 14, "Pleiades DR2 data")
ax[0].set_title("Dias+2021")

# collect legend data from all subplots
handles, labels = ax[0].get_legend_handles_labels()
entries_to_skip = 1
handles_new = handles[entries_to_skip:]
labels_new = labels[entries_to_skip:]
ax[0].legend(handles_new, labels_new, loc="upper right", bbox_to_anchor=(1, 0.85))

# Subplot 2
ax[1].scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="data")
ax[1].plot(Bossini["BP-RP"][start:stop], Bossini["Gmag"][start:stop], color="#66a61e", label="86 Myr")
ax[1].set_title("Bossini+2019")
handles, labels = ax[1].get_legend_handles_labels()
entries_to_skip = 1
handles_new = handles[entries_to_skip:]
labels_new = labels[entries_to_skip:]
ax[1].legend(handles_new, labels_new, loc="upper right", bbox_to_anchor=(1, 0.85))

# Subplot 3
ax[2].scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD, label="data")
ax[2].plot(CG["BP-RP"][start:stop], CG["Gmag"][start:stop], color="#e6ab02", label="78 Myr")
ax[2].set_title("Cantat-Gaudin+2020")
handles, labels = ax[2].get_legend_handles_labels()
entries_to_skip = 1
handles_new = handles[entries_to_skip:]
labels_new = labels[entries_to_skip:]
ax[2].legend(handles_new, labels_new, loc="upper right", bbox_to_anchor=(1, 0.85))

ylim = ax[2].get_ylim()
xlim = ax[2].get_xlim()

# flip Y-axes for all supblots
for a in ax:
    a.set_ylim(ylim[1], ylim[0])

ax[0].set_ylabel(r"M$_{\mathrm{G}}$", labelpad=1)
ax[1].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$", labelpad=1)

plt.show()
if save_plot:
    fig_poster.savefig(output_path + "Pleiades_comparison_no_hints_c2.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Interactive solution plot
if interactive:
    import plotly.graph_objs as go
    from plotly.offline import plot
    from matplotlib import colors

# Extract RGB values from colormap and convert to hex
hex_values = []

# build a colormap
cm = plt.cm.ScalarMappable(cmap=OC.kwargs_CMD["cmap"]).to_rgba(OC.kwargs_CMD["c"])
col_hex = [colors.rgb2hex(c) for c in cm]

# initialize figure
fig = go.Figure()
fig.update_xaxes(range=[xlim[0], xlim[1]])
fig.update_yaxes(range=[lim[1], lim[0]])
fig.update_layout(width=750, height=850)
fig.update_layout(template='plotly_white')

# scatter Pleiades data
scatter = go.Scatter(x=OC.density_x, y=OC.density_y, mode="markers",
                     marker=dict(size=5, color=col_hex, cmin=1, symbol="circle"), visible=True,
                     name="Pleiades DR2 data")
fig.add_trace(scatter)

# add the three model isochrones
for source, label, col in zip([Dias, Bossini, CG],
                              ["131 Myr (Dias+2021)", "86 Myr (Bossini+2019)", "78 Myr (Cantat-Gaudin+2020)"],
                              ["#e7298a", "#66a61e", "#e6ab02"]):
    trace = go.Scatter(x=source["BP-RP"][start:stop],
                       y=source["Gmag"][start:stop],
                       name=label,
                       visible=True,  # make the first line visible by default
                       line=dict(color=col, width=2))
    fig.add_trace(trace)

# background layout
fig.update_layout(
    plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
    xaxis=dict(gridcolor='white'),  # Set the x-axis grid line color to white
    yaxis=dict(gridcolor='white'),  # Set the y-axis grid line color to white
)

# ax and title layout
fig.update_layout(
    xaxis_title="BP-RP",
    yaxis_title="absolute G magnitude",
    autosize=True,
    margin=dict(l=50, r=50, t=50, b=50),
    title={
        'text': 'How old are the Pleiades?',
        'x': 0.5,
        'xanchor': 'center'}
)

plot(fig, filename=output_path + 'Pleiades_ages.html')
# ----------------------------------------------------------------------------------------------------------------------
