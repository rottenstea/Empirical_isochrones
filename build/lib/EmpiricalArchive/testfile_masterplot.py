import matplotlib.pyplot as plt
import seaborn as sns

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.My_tools.plotting_essentials import Archive_plot_2D
from EmpiricalArchive.Extraction.Empirical_iso_reader import merged_BPRP as BPRP_data
from EmpiricalArchive.Extraction.Empirical_iso_reader import merged_BPG as BPG_data
from EmpiricalArchive.Extraction.Empirical_iso_reader import merged_GRP as GRP_data
# ----------------------------------------------------------------------------------------------------------------------
# STANDARD PLOT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()

sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 10

age_col = "ref_age"
a = plt.get_cmap("YlGnBu_r")
norm = plt.Normalize(BPRP_data[age_col].min(), BPRP_data[age_col].max())
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])

dict1 = {'palette': a, 'hue_norm': norm}

age_low = 5
age_high = 11

plotting_dict = {'x': 'm_x', 'y': 'm_y', 'legend': False, 'sort': False, 'lw': 1, 'units': "Cluster_id",
                 'estimator': None} | dict1


fig = Archive_plot_2D([BPRP_data, BPG_data, GRP_data], [age_col, age_high, age_low], plotting_dict, sm)

fig.show()