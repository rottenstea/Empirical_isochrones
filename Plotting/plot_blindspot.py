import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from My_tools import my_utility
from Extraction.Empirical_iso_reader import merged_BPRP, merged_BPG, merged_GRP
from Extraction.pre_processing import cluster_df_list

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
# PARSEC isochrones for illustrating the blindspot
blindspot_parsec = pd.read_csv(
    "/Users/alena/PycharmProjects/PaperI/data/Isochrones/PARSEC_isochrones/Bindspot/Blindspot_PARSEC.csv",
    usecols=["logAge", "linAge", "Gmag", "G_BPmag", "G_RPmag"], comment="#")
blindspot_parsec["BP_RP"] = blindspot_parsec["G_BPmag"] - blindspot_parsec["G_RPmag"]
blindspot_parsec["BP_G"] = blindspot_parsec["G_BPmag"] - blindspot_parsec["Gmag"]
blindspot_parsec["G_RP"] = blindspot_parsec["Gmag"] - blindspot_parsec["G_RPmag"]

Archive_df = pd.concat([cluster_df_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# Only blindspot

# Colormap
a = plt.get_cmap("YlGnBu_r")
norm = plt.Normalize(Archive_df['ref_age'].min(), Archive_df['ref_age'].max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

theory_fig, ax = plt.subplots(1, 3, figsize=(7.24551, 4.), sharey="row")
plt.subplots_adjust(left=0.075, top=0.99, bottom=0.105, wspace=0.05, right=0.87)

# Plot the theoretical isochrone for all three passbands for each age in the PARSEC file
ages = blindspot_parsec["linAge"].unique()[1:]
for j, age in enumerate(ages):
    color = a(j / 15)
    df = blindspot_parsec[blindspot_parsec["linAge"] == age]
    ax[0].plot(df["BP_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)
    ax[1].plot(df["BP_G"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)
    ax[2].plot(df["G_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)

ax[0].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)")
ax[1].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}$ (mag)")
ax[2].set_xlabel(r"$\mathrm{G} - \mathrm{G}_{\mathrm{RP}}$ (mag)")

ax[0].set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$ (mag)", labelpad=1)
plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1), title="Ages (Myr)")
ax[0].set_ylim(15, -4)

# Indicate upper and lower MS
for axis in ax:
    xlim = axis.get_xlim()
    axis.hlines(y=0, xmin=xlim[0], xmax=xlim[1], lw=1, ls="--", color="darkslategray", alpha=0.75)
    axis.hlines(y=10, xmin=xlim[0], xmax=xlim[1], lw=1, ls="-.", color="darkslategray", alpha=0.75)

theory_fig.show()
if save_plot:
    theory_fig.savefig(output_path + "Blindspot_Gaia_limits.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# Blindspot comparison figure
a = plt.get_cmap("YlGnBu_r")
# Norm now created from archive clusters
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

# define age limits for blindspot clusters
ages = blindspot_parsec["linAge"].unique()[1:]
age_low = np.log10(ages[0] * 1e6)
age_up = np.log10(ages[-1] * 1e6)

comp_fig, ax = plt.subplots(1, 3, figsize=(7.24551, 4.4), sharey="row")
plt.subplots_adjust(left=0.08, bottom=0.0, top=0.99, right=0.99, wspace=0.1)

# Subplot 1
BPRP = sns.lineplot(data=merged_BPRP[(merged_BPRP["ref_age"] >= age_low) & (merged_BPRP["ref_age"] <= age_up)], x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$ (mag)",
                                                                            ylabel=r"$\mathrm{M}_{\mathrm{G}}$ (mag)")
ax[0].set_xlim(-1.9, 5)
ax[0].set_ylim(15, -3.5)

# Subplot 2
BPG = sns.lineplot(data=merged_BPG[(merged_BPG["ref_age"] >= age_low) & (merged_BPG["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                  r"\mathrm{G}$ (mag)")
ax[1].set_xlim(-1.1, 3.2)
ax[1].set_ylim(15, -4.1)

# Subplot 3
RPG = sns.lineplot(data=merged_GRP[(merged_GRP["ref_age"] >= age_low) & (merged_GRP["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[2], estimator=None).set(xlabel=r"$\mathrm{G} - "
                                                                                  r"\mathrm{G}_{\mathrm{RP}}$ (mag)")
ax[2].set_xlim(-0.5, 1.8)
ax[2].set_ylim(15, -4.2)

# Colorbar
c = comp_fig.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35, pad=0.12)
cax = c.ax
cax.tick_params(labelsize=10)
cax.text(6.65, 0.3, 'log age')

# Outline the  blindspot edge isochrones (PARSEC) 
for j, age in enumerate(ages):
    if j == 0 or j == 10:
        df = blindspot_parsec[blindspot_parsec["linAge"] == age]
        ax[0].plot(df["BP_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=1, alpha=0.75, label=age)
        ax[1].plot(df["BP_G"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=1, alpha=0.75, label=age)
        ax[2].plot(df["G_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)

# Indicate lower MS
for axis in ax:
    xlim = axis.get_xlim()
    axis.hlines(y=10, xmin=xlim[0], xmax=xlim[1], ls="-.", color="darkslategray", alpha=0.75, lw=1)

comp_fig.show()
if save_plot:
    comp_fig.savefig(output_path + "Blindspot_vs_empirical.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
