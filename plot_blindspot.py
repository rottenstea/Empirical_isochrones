import my_utility
from Empirical_iso_reader import merged_BPRP, merged_BPG, merged_GRP
from pre_processing import cluster_df_list

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

output_path = my_utility.set_output_path()
save_plot = False

blindspot_parsec = pd.read_csv(
    "/Users/alena/PycharmProjects/PaperI/data/Isochrones/PARSEC_isochrones/Bindspot/Blindspot_PARSEC.csv",
    usecols=["logAge", "linAge", "Gmag", "G_BPmag", "G_RPmag"], comment="#")
blindspot_parsec["BP_RP"] = blindspot_parsec["G_BPmag"] - blindspot_parsec["G_RPmag"]
blindspot_parsec["BP_G"] = blindspot_parsec["G_BPmag"] - blindspot_parsec["Gmag"]
blindspot_parsec["G_RP"] = blindspot_parsec["Gmag"] - blindspot_parsec["G_RPmag"]

ages = blindspot_parsec["linAge"].unique()[1:]

# Blindspot comparison figure
# -------------------------------------------

a = plt.get_cmap("YlGnBu_r")
sns.set_style("darkgrid")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

age_low = np.log10(ages[0] * 1e6)
age_up = np.log10(ages[-1] * 1e6)

fig_2D, ax = plt.subplots(1, 3, figsize=(8, 5), layout="constrained", sharey="row")

BPRP = sns.lineplot(data=merged_BPRP[(merged_BPRP["ref_age"] >= age_low) & (merged_BPRP["ref_age"] <= age_up)], x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                   r"\mathrm{G}_{\mathrm{RP}}$",
                                                                            ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[0].set_xlim(-1.9, 5)
ax[0].set_ylim(15, -3.5)

BPG = sns.lineplot(data=merged_BPG[(merged_BPG["ref_age"] >= age_low) & (merged_BPG["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                  r"\mathrm{G}$")
ax[1].set_xlim(-1.1, 3.2)
ax[1].set_ylim(15, -4.1)

RPG = sns.lineplot(data=merged_GRP[(merged_GRP["ref_age"] >= age_low) & (merged_GRP["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1, units="Cluster_id", ax=ax[2], estimator=None).set(xlabel=r"$\mathrm{G} - "
                                                                                  r"\mathrm{G}_{\mathrm{RP}}$")
ax[2].set_xlim(-0.5, 1.8)
ax[2].set_ylim(15, -4.2)

c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
cax = c.ax
cax.text(6.65, 0.3, 'log age')

for j, age in enumerate(ages):
    if j == 0 or j == 10:
        df = blindspot_parsec[blindspot_parsec["linAge"] == age]
        ax[0].plot(df["BP_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)
        ax[1].plot(df["BP_G"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)
        ax[2].plot(df["G_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)

# fig_2D.show()
if save_plot:
    fig_2D.savefig(output_path + "Blindspot_Summary.pdf", dpi=600)

# Only blindspot
# ------------------------------

Archive_df = pd.concat([cluster_df_list[i] for i in [0, 2, 3, 4, 5]], axis=0)

a = plt.get_cmap("YlGnBu_r")
sns.set_style("darkgrid")
norm = plt.Normalize(Archive_df['ref_age'].min(), Archive_df['ref_age'].max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

fig_3, ax = plt.subplots(1, 3, figsize=(8, 4), sharey="row")
sns.set_style("darkgrid")

ages = blindspot_parsec["linAge"].unique()[1:]
for j, age in enumerate(ages):
    color = a(j / 15)
    df = blindspot_parsec[blindspot_parsec["linAge"] == age]
    ax[0].plot(df["BP_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)
    ax[1].plot(df["BP_G"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)
    ax[2].plot(df["G_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color=color, label=age)

ax[0].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
ax[1].set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}$")
ax[2].set_xlabel(r"$\mathrm{G} - \mathrm{G}_{\mathrm{RP}}$")

ax[0].set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$")
plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
ax[0].set_ylim(15, -4)
plt.subplots_adjust(left=0.08, top=0.98, bottom=0.11, wspace=0.1, right=0.86)

# fig_3.show()
if save_plot:
    fig_3.savefig(output_path + "Blindspot_Gaia.pdf", dpi=600)
