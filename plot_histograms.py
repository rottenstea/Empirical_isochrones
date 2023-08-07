import my_utility
from pre_processing import cluster_df_list, cluster_name_list

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from astropy.visualization import hist as astrohist
import matplotlib.gridspec as gridspec

output_path = my_utility.set_output_path()
savefig = True

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

CI_df, CII_df, CIII_df, AOI_df, AOII_df, AOIII_df, AOIV_df = cluster_df_list
CI_names, CII_names, CIII_names, AOI_names, AOII_names, AOIII_names, AOIV_names = cluster_name_list

alph = 1

CI_df.drop_duplicates("Cluster_id", inplace=True)

logage_B = CI_df["age_B"]
logage_C = CI_df["age_C"]
logage_D = CI_df["age_D"]

age_B = 10 ** (CI_df["age_B"]) / 1e6
age_C = 10 ** (CI_df["age_C"]) / 1e6
age_D = 10 ** (CI_df["age_D"]) / 1e6

fig = plt.figure(figsize=(10, 7))

# Catalog I
ax11 = plt.subplot2grid((2, 3), (0, 0))
ax12 = plt.subplot2grid((2, 3), (0, 1), sharex=ax11, sharey=ax11)
ax13 = plt.subplot2grid((2, 3), (0, 2), sharex=ax11, sharey=ax11)

ax21 = plt.subplot2grid((2, 3), (1, 0))
ax22 = plt.subplot2grid((2, 3), (1, 1), sharex=ax21, sharey=ax21)
ax23 = plt.subplot2grid((2, 3), (1, 2), sharex=ax21, sharey=ax21)

# -----------------------------------------------------------------------
ax11.hist(x=age_D, bins=len(age_D), histtype='step', alpha=alph)
ax11.hist(x=age_C, bins=len(age_C), histtype='step', alpha=alph)
ax11.hist(x=age_B, bins=len(age_B), alpha=alph)
ax11.set_title("Bossini $et. al.$ 2019")

ax12.hist(x=age_D, histtype='step', alpha=alph)
ax12.hist(x=age_C, alpha=alph)
ax12.hist(x=age_B, histtype='step', alpha=alph)
ax12.set_title("Cantat-Gaudin ${et. al.}$ 2020a")

ax13.hist(x=age_D, alpha=alph)
ax13.hist(x=age_C, histtype='step', alpha=alph)
ax13.hist(x=age_B, histtype='step', alpha=alph)
ax13.set_title("Dias $et. al.$ 2020")
# ----------------------------------------------------------------------
ax21.hist(x=logage_D, bins=len(logage_D), histtype='step', alpha=alph)
ax21.hist(x=logage_C, bins=len(logage_C), histtype='step', alpha=alph)
ax21.hist(x=logage_B, bins=len(logage_B), alpha=alph)
ax21.set_title("Bossini $et. al.$ 2019")

ax22.hist(x=logage_D, histtype='step', alpha=alph)
ax22.hist(x=logage_C, alpha=alph)
ax22.hist(x=logage_B, histtype='step', alpha=alph)
ax22.set_title("Cantat-Gaudin ${et. al.}$ 2020a")

ax23.hist(x=logage_D, alpha=alph)
ax23.hist(x=logage_C, histtype='step', alpha=alph)
ax23.hist(x=logage_B, histtype='step', alpha=alph)
ax23.set_title("Dias $et. al.$ 2020")
# -----------------------------------------------------------------------
plt.subplots_adjust(wspace=0.3, hspace=0.4)
#plt.show()

#if savefig:
#    plt.savefig(output_path+"Age_histograms.pdf", dpi=500)

age_D_arr, logage_D_arr = age_D.to_numpy(), logage_D.to_numpy()
age_D_arr, logage_D_arr = age_D_arr[~np.isnan(age_D_arr)], logage_D_arr[~np.isnan(logage_D_arr)]

age_B_arr, logage_B_arr = age_B.to_numpy(), logage_B.to_numpy()
age_B_arr, logage_B_arr = age_B_arr[~np.isnan(age_B_arr)], logage_B_arr[~np.isnan(logage_B_arr)]

age_C_arr, logage_C_arr = age_C.to_numpy(), logage_C.to_numpy()
age_C_arr, logage_C_arr = age_C_arr[~np.isnan(age_C_arr)], logage_C_arr[~np.isnan(logage_C_arr)]
'''
fig2 = plt.figure(figsize=(10, 7))

# Catalog I
ax11 = plt.subplot2grid((2, 3), (0, 0))
ax12 = plt.subplot2grid((2, 3), (0, 1), sharex=ax11, sharey=ax11)
ax13 = plt.subplot2grid((2, 3), (0, 2), sharex=ax11, sharey=ax11)

ax21 = plt.subplot2grid((2, 3), (1, 0))
ax22 = plt.subplot2grid((2, 3), (1, 1), sharex=ax21, sharey=ax21)
ax23 = plt.subplot2grid((2, 3), (1, 2), sharex=ax21, sharey=ax21)

fig2.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for array, title, ax in zip([age_B_arr, age_C_arr, age_D_arr, logage_B_arr, logage_C_arr, logage_D_arr],
                           ["Bossini $et. al.$ 2019", "Cantat-Gaudin ${et. al.}$ 2020a", "Dias $et. al.$ 2020",
                            "Bossini $et. al.$ 2019", "Cantat-Gaudin ${et. al.}$ 2020a", "Dias $et. al.$ 2020"],
                           [ax11, ax12, ax13, ax21, ax22, ax23]):
    # plot a standard histogram in the background, with alpha transparency
    astrohist(array, bins=len(array), histtype='stepfilled', ax = ax,
            alpha=0.2, density=True, label='standard histogram')

    # plot an adaptive-width histogram on top
    astrohist(array, bins="blocks", ax=ax, color='black', max_bins=len(array),
         histtype='step', density=True, label=title)

    ax.legend(prop=dict(size=12))
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    #ax.set_title(title)

plt.show()

'''
fig2 = plt.figure(figsize=(10, 4))

# Catalog I
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

#plt.show()

#fig2.savefig(output_path + "logage_hist_knuth_C1.pdf", dpi=500)

fig3 = plt.figure(figsize=(6, 4))

# Catalog I
ax11 = plt.subplot2grid((1, 3), (0, 0))
ax12 = plt.subplot2grid((1, 3), (0, 1), sharey=ax11)
ax13 = plt.subplot2grid((1, 3), (0, 2), sharey=ax11)

#if savefig:
#    fig2.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for array, title, ax in zip([age_B_arr, age_C_arr, age_D_arr],
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

#plt.show()
#if savefig:
 #   fig3.savefig(output_path + "age_hist_knuth_C1.pdf", dpi=500)

Archive_df = pd.concat([cluster_df_list[i] for i in [0, 2, 3, 4, 5]], axis=0)
Archive_df.drop_duplicates("Cluster_id", inplace=True)

logages = Archive_df["ref_age"].to_numpy()
ages = 10 ** logages / 1e6

ages, logages = ages[~np.isnan(ages)], logages[~np.isnan(logages)]

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

fig4 = plt.figure(figsize=(3.54399, 2.5))
colors_f4 = ['#a1dab4', '#41b6c4', '#225ea8']

ax = plt.subplot2grid((1, 1), (0, 0))

plt.subplots_adjust(left = 0.125,right = 0.98, top=.99, bottom=0.171)


#astrohist(logages, bins=len(logages), histtype='stepfilled', ax=ax, density=False,
 #         alpha=0.6, label='true distribution', color=colors_f4[2], edgecolor = colors_f4[2] )  # , edgecolor = colors_f4[2])

# astrohist(logages, bins="blocks", ax=ax, color=colors_f4[0], alpha = 1, lw = 2,
#             histtype='step', label="Blocks", density = False)

astrohist(logages, bins="knuth", ax=ax, color=colors_f4[2], alpha=1, lw=2,
          histtype='step',  density=False)

ax.set_xlabel('log age')
ax.set_ylabel('Count')
#ax.legend(loc="upper right")

fig4.show()

if savefig:
    fig4.savefig(output_path + "Histogram_logage.pdf", dpi=600)

fig5 = plt.figure(figsize=(6, 3))
colors_f5 = ['#a1dab4', '#41b6c4', '#225ea8']

spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig4)

ax0 = fig5.add_subplot(spec[0, :2])
ax11 = fig5.add_subplot(spec[0, 2])
ax12 = fig5.add_subplot(spec[0, 3])

ax11.set_xlim(0, 100)
ax12.set_xlim(100, 3000)
ax11.spines["right"].set_visible(False)
ax12.spines["left"].set_visible(False)
ax12.set_yticklabels([])

plt.subplots_adjust(left=0.05, wspace=0.35, top=0.95, bottom=0.15)

for array, title, ax in zip([logages, ages, ages],
                            ["logages", "ages", "ages"],
                            [ax0, ax11, ax12]):
    # plot a standard histogram in the background, with alpha transparency
    astrohist(array, bins=len(array), histtype='stepfilled', ax=ax, density=True,
              alpha=0.7, label='true dist', color=colors_f4[2], edgecolor=colors_f4[2])

    # plot an adaptive-width histogram on top

    # astrohist(array, bins="blocks", ax=ax, color=colors_f4[0], alpha = 1, lw = 2,
    #          histtype='step', label="Blocks", density = True)

    # astrohist(array, bins="knuth", ax=ax, color=colors_f4[1], alpha = 1, lw = 2,
    #          histtype='step', label="Knuth", density = True)

    # ax.set_title(title)

# ax11.set_xlabel('log age')
# ax12.set_xlabel('age / Myr')
# ax11.set_ylabel('Density')
# ax12.legend(prop=dict(size=12))
