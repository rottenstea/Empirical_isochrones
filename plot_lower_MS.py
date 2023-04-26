import my_utility
from Empirical_iso_reader import build_empirical_df, general_kwargs
from plot_blindspot import blindspot_parsec
from Empirical_iso_reader import merged_BPRP

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def lower_MS_filter(dataf_list):
    return [entry for entry in dataf_list if any(entry["m_y"] >= 10)]


output_path = my_utility.set_output_path()
save_plot = False

# Lower Main sequence isochrones
# -------------------------------------
name_keys = ["G_BPRP_nboot_1000", "G_BPG_nboot_1000", "G_GRP_nboot_1000"]
merged = []
for key in name_keys:
    merged_df = build_empirical_df(filename_key=key, filter_function=lower_MS_filter, **general_kwargs)
    merged.append(merged_df)
m_BPRP, m_BPG, m_GRP = merged

ages = blindspot_parsec["linAge"].unique()[1:]

a = plt.get_cmap("YlGnBu_r")
sns.set_style("darkgrid")
norm = plt.Normalize(merged_BPRP.ref_age.min(), merged_BPRP.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

age_low = np.log10(100 * 1e6)
age_up = np.log10(600 * 1e6)

fig_4, ax = plt.subplots(3, 1, figsize=(4, 10), layout="constrained", sharey="row")

BPRP = sns.lineplot(data=m_BPRP[(m_BPRP["ref_age"] >= age_low) & (m_BPRP["ref_age"] <= age_up)], x="m_x",
                    y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                    lw=1.2, units="Cluster_id", ax=ax[0], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                     r"\mathrm{G}_{\mathrm{RP}}$",
                                                                              ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[0].set_xlim(1, 5)
ax[0].set_ylim(15, 6.5)

BPG = sns.lineplot(data=m_BPG[(m_BPG["ref_age"] >= age_low) & (m_BPG["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1.2, units="Cluster_id", ax=ax[1], estimator=None).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - "
                                                                                    r"\mathrm{G}$",
                                                                             ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[1].set_xlim(0.5, 3.2)
ax[1].set_ylim(15, 6.5)

RPG = sns.lineplot(data=m_GRP[(m_GRP["ref_age"] >= age_low) & (m_GRP["ref_age"] <= age_up)], x="m_x",
                   y="m_y", hue="ref_age", palette=a, hue_norm=norm, legend=False, sort=False,
                   lw=1.2, units="Cluster_id", ax=ax[2], estimator=None).set(xlabel=r"$\mathrm{G} - "
                                                                                    r"\mathrm{G}_{\mathrm{RP}}$",
                                                                             ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
ax[2].set_xlim(0.5, 1.8)
ax[2].set_ylim(15, 6.5)

# c = fig_2D.colorbar(sm, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35)
# cax = c.ax
# cax.text(6.65, 0.3, 'log age')

for j, age in enumerate(ages):
    if j == 0 or j == 10:
        df = blindspot_parsec[blindspot_parsec["linAge"] == age]
        ax[0].plot(df["BP_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)
        ax[1].plot(df["BP_G"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)
        ax[2].plot(df["G_RP"].iloc[3:120], df["Gmag"].iloc[3:120], color="black", lw=0.75, label=age)

fig_4.show()
if save_plot:
    fig_4.savefig(output_path + "Lower_MS.pdf", dpi=600)
