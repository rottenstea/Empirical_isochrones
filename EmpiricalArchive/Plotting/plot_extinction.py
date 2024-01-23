import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from EmpiricalArchive.Extraction.pre_processing import cluster_df_list, cluster_name_list

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
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# get catalogs
CI_df, CII_df, CIII_df, AOI_df, AOII_df, AOIII_df, AOIV_df = cluster_df_list
CI_names, CII_names, CIII_names, AOI_names, AOII_names, AOIII_names, AOIV_names = cluster_name_list

# Extinction vectors and positions for CI DR2
extinction_DR2 = [0.5290999999999997, 1]
arrow_pos1 = [3.25, 7.]
arrow_pos3 = [3.5, 7]

# Prep DR2 data for Catalog I
CMD_parameters = ["Gmag_DR2", "BP-RP_DR2"]
parallax = "plx_DR2"
CI_df = CI_df[~np.isnan(CI_df.excess)]
CI_distance = 1000 / CI_df[parallax].to_numpy()
CI_mag, CI_color_ax = CI_df[CMD_parameters[0]].to_numpy(), CI_df[CMD_parameters[1]].to_numpy()
CI_abs_mag = (CI_mag - 5 * np.log10(CI_distance) + 5)
CI_arr = np.stack([CI_color_ax, CI_abs_mag], axis=1)
CI_cleaned_arr = CI_arr[~np.isnan(CI_arr).any(axis=1), :]
CI_sorted_arr = CI_cleaned_arr[CI_cleaned_arr[:, 1].argsort()]
rho_x_C1, rho_y_C1, kwargs_C1 = CMD_density_design(CI_sorted_arr, density_plot=False)
kwargs_C1["s"] = 20

# Prep DR2 data for Catalog III
CIII_distance = 1000 / CIII_df[parallax].to_numpy()
CIII_mag, CIII_color_ax = CIII_df[CMD_parameters[0]].to_numpy(), CIII_df[CMD_parameters[1]].to_numpy()
CIII_abs_mag = (CIII_mag - 5 * np.log10(CIII_distance) + 5)
CIII_arr = np.stack([CIII_color_ax, CIII_abs_mag], axis=1)
CIII_cleaned_arr = CIII_arr[~np.isnan(CIII_arr).any(axis=1), :]
CIII_sorted_arr = CIII_cleaned_arr[CIII_cleaned_arr[:, 1].argsort()]
rho_x_C3, rho_y_C3, kwargs_C3 = CMD_density_design(CIII_sorted_arr, density_plot=False)
kwargs_C3["s"] = 20
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(3.54399, 3.2))
plt.subplots_adjust(left=0.123, right=0.99, top=.93, bottom=.125, wspace=.05)
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# Subplot 1
ax1.scatter(rho_x_C1, rho_y_C1, **kwargs_C1)
ylim = ax1.get_ylim()
ax1.set_ylim(ylim[1], ylim[0])

ax1.annotate("", xy=(arrow_pos1[0] + extinction_DR2[0], arrow_pos1[1] + extinction_DR2[1]),
             xytext=(arrow_pos1[0], arrow_pos1[1]),
             arrowprops=dict(arrowstyle="->", color="k"))
ax1.text(arrow_pos1[0] - 0.25, arrow_pos1[1] + 0.5, r"$\mathrm{A}_{\mathrm{G}}$=1", size=10,
         rotation=-44)

ax1.set_title("Catalog I", fontsize=10)
ax1.set_ylabel(r"M$_{\mathrm{G}}$ (mag)", labelpad=1)
ax1.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)", labelpad=2, x=1)

# Subplot 2
ax2.scatter(rho_x_C3, rho_y_C3, **kwargs_C3)
ax2.set_ylim(ylim[1], ylim[0])

ax2.annotate("", xy=(arrow_pos3[0] + extinction_DR2[0], arrow_pos3[1] + extinction_DR2[1]),
             xytext=(arrow_pos3[0], arrow_pos3[1]),
             arrowprops=dict(arrowstyle="->", color="k"))
ax2.text(arrow_pos3[0] - 0.25, arrow_pos3[1] + 0.5, r"$\mathrm{A}_{\mathrm{G}}$=1", size=10,
         rotation=-44)

ax2.set_title("Catalog III", fontsize=10)
ax2.set_yticklabels([])

fig.show()
if save_plot:
    fig.savefig(output_path + "Extinction_plots.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
