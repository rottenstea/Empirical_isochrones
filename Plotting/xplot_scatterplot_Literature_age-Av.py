import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot

from My_tools import my_utility
from Extraction.pre_processing import cluster_df_list, cluster_name_list

# ----------------------------------------------------------------------------------------------------------------------
# Set output path to the Coding-logfile
output_path = my_utility.set_output_path()
# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
unique_entry_list = []

# drop CII clusters to avoid duplicates
cluster_name_list = [cluster_name_list[i] for i in [0, 2, 3, 4, 5]]
cluster_df_list = [cluster_df_list[i] for i in [0, 2, 3, 4, 5]]

for cluster_df in cluster_df_list:
    unique_entries = cluster_df.drop_duplicates(subset="Cluster_id")
    unique_entry_list.append(unique_entries)

Archive = np.concatenate(cluster_name_list, axis=0)
Archive_df = pd.concat(unique_entry_list, axis=0)

# If only subsamples are of interest
# CI = Archive_df[Archive_df["catalog"]== 1]
# CIII = Archive_df[Archive_df["catalog"]== 3]
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# Extinction
f2_av = px.scatter(Archive_df, x="Cluster_id", y=["av_C", "av_B", "av_D"], hover_data=["ref_age"])
plot(f2_av, filename=output_path + 'Extinction_scatter.html')
# ----------------------------------------------------------------------------------------------------------------------
# Age
f2 = px.scatter(Archive_df, x="Cluster_id", y=["age_C", "age_B", "age_D"])
plot(f2, filename=output_path + 'Age_scatter.html')
# ----------------------------------------------------------------------------------------------------------------------
