import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.visualization import hist as astrohist

from My_tools import my_utility
from Extraction.pre_processing import cluster_df_list, cluster_name_list

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
xplot = False
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Collect dataframes
CI_df, CII_df, CIII_df, AOI_df, AOII_df, AOIII_df, AOIV_df = cluster_df_list
CI_names, CII_names, CIII_names, AOI_names, AOII_names, AOIII_names, AOIV_names = cluster_name_list
CI_df.drop_duplicates("Cluster_id", inplace=True)

# define the different age estimates
logage_B = CI_df["age_B"]
logage_C = CI_df["age_C"]
logage_D = CI_df["age_D"]

age_B = 10 ** (CI_df["age_B"]) / 1e6
age_C = 10 ** (CI_df["age_C"]) / 1e6
age_D = 10 ** (CI_df["age_D"]) / 1e6
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# Simple histogram for paper
Archive_df = pd.concat([cluster_df_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
Archive_df.drop_duplicates("Cluster_id", inplace=True)

# use reference ages
logages = Archive_df["ref_age"].to_numpy()
ages = 10 ** logages / 1e6

# remove NaNs
ages, logages = ages[~np.isnan(ages)], logages[~np.isnan(logages)]

hist_fig = plt.figure(figsize=(3.54399, 2.5))
colors_f4 = ['#a1dab4', '#41b6c4', '#225ea8']
ax = plt.subplot2grid((1, 1), (0, 0))
plt.subplots_adjust(left=0.125, right=0.98, top=.99, bottom=0.171)

astrohist(logages, bins="knuth", ax=ax, color=colors_f4[2], alpha=1, lw=2,
          histtype='step', density=False)

ax.set_xlabel('log age')
ax.set_ylabel('Count')
# ax.legend(loc="upper right")

hist_fig.show()
if save_plot:
    hist_fig.savefig(output_path + "Histogram_logage.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
# XPLOT
# ----------------------------------------------------------------------------------------------------------------------
if xplot:
    # Literature age comparison Catalog 1
    fig = plt.figure(figsize=(10, 7))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Age row
    ax11 = plt.subplot2grid((2, 3), (0, 0))
    ax12 = plt.subplot2grid((2, 3), (0, 1), sharex=ax11, sharey=ax11)
    ax13 = plt.subplot2grid((2, 3), (0, 2), sharex=ax11, sharey=ax11)

    # Logage row
    ax21 = plt.subplot2grid((2, 3), (1, 0))
    ax22 = plt.subplot2grid((2, 3), (1, 1), sharex=ax21, sharey=ax21)
    ax23 = plt.subplot2grid((2, 3), (1, 2), sharex=ax21, sharey=ax21)

    # Suplot 1
    ax11.hist(x=age_D, bins=len(age_D), histtype='step')
    ax11.hist(x=age_C, bins=len(age_C), histtype='step')
    ax11.hist(x=age_B, bins=len(age_B))
    ax11.set_title("Bossini $et. al.$ 2019")

    # Subplot 2
    ax12.hist(x=age_D, histtype='step')
    ax12.hist(x=age_C)
    ax12.hist(x=age_B, histtype='step')
    ax12.set_title("Cantat-Gaudin ${et. al.}$ 2020a")

    # Subplot 3
    ax13.hist(x=age_D)
    ax13.hist(x=age_C, histtype='step')
    ax13.hist(x=age_B, histtype='step')
    ax13.set_title("Dias $et. al.$ 2020")

    # Subplot 4
    ax21.hist(x=logage_D, bins=len(logage_D), histtype='step')
    ax21.hist(x=logage_C, bins=len(logage_C), histtype='step')
    ax21.hist(x=logage_B, bins=len(logage_B))
    ax21.set_title("Bossini $et. al.$ 2019")

    # Subplot 5
    ax22.hist(x=logage_D, histtype='step')
    ax22.hist(x=logage_C)
    ax22.hist(x=logage_B, histtype='step')
    ax22.set_title("Cantat-Gaudin ${et. al.}$ 2020a")

    # Subplot 6
    ax23.hist(x=logage_D)
    ax23.hist(x=logage_C, histtype='step')
    ax23.hist(x=logage_B, histtype='step')
    ax23.set_title("Dias $et. al.$ 2020")

    plt.show()
    if save_plot:
        plt.savefig(output_path + "Age_histograms.pdf", dpi=600)
    # ----------------------------------------------------------------------------------------------------------------------
    # Knuth histograms for all literature sources

    # Clean NaNs
    age_D_arr, logage_D_arr = age_D.to_numpy(), logage_D.to_numpy()
    age_D_arr, logage_D_arr = age_D_arr[~np.isnan(age_D_arr)], logage_D_arr[~np.isnan(logage_D_arr)]

    age_B_arr, logage_B_arr = age_B.to_numpy(), logage_B.to_numpy()
    age_B_arr, logage_B_arr = age_B_arr[~np.isnan(age_B_arr)], logage_B_arr[~np.isnan(logage_B_arr)]

    age_C_arr, logage_C_arr = age_C.to_numpy(), logage_C.to_numpy()
    age_C_arr, logage_C_arr = age_C_arr[~np.isnan(age_C_arr)], logage_C_arr[~np.isnan(logage_C_arr)]

    fig2 = plt.figure(figsize=(10, 4))
    ax11 = plt.subplot2grid((1, 3), (0, 0))
    ax12 = plt.subplot2grid((1, 3), (0, 1), sharey=ax11)
    ax13 = plt.subplot2grid((1, 3), (0, 2), sharey=ax11)
    fig2.subplots_adjust(left=0.14, right=0.95, bottom=0.16)

    for array, title, ax in zip([logage_B_arr, logage_C_arr, logage_D_arr],
                                ["Bossini $et. al.$ 2019", "Cantat-Gaudin ${et. al.}$ 2020a", "Dias $et. al.$ 2020"],
                                [ax11, ax12, ax13]):
        # plot a standard histogram in the background, with alpha transparency
        astrohist(array, bins=len(array), histtype='stepfilled', ax=ax,
                  alpha=0.4, label='true dist')

        # plot an adaptive-width histogram on top
        astrohist(array, bins="knuth", ax=ax, color='black',
                  histtype='step', label="knuth")

        ax.set_xlabel('log age')
        ax.set_title(title)

    ax11.set_ylabel('Count')
    ax13.legend(prop=dict(size=12))

    plt.show()
    if save_plot:
        fig2.savefig(output_path + "logage_hist_knuth_C1.pdf", dpi=600)
# ----------------------------------------------------------------------------------------------------------------------
