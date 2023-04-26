import my_utility
from Classfile import *

from pre_processing import cluster_df_list, cluster_name_list, case_study_names, case_study_dfs

main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
empirical_iso_path = "/Users/alena/PycharmProjects/PaperI/data/Isochrones/Empirical/"
output_path = my_utility.set_output_path()


print("CATALOGS")
for ncat, catalog in enumerate(cluster_name_list):
    print("Catalog", ncat+1)
    N = np.empty(shape=len(catalog))
    mean_d = np.empty(shape=len(catalog))
    for j, cluster in enumerate(catalog):
        OC = star_cluster(cluster, cluster_df_list[ncat])
        mean_d[j] = np.mean(OC.distance)
        N[j] = OC.Nstars

    print("Distance")
    print(catalog[np.argmin(mean_d)], "min:", np.min(mean_d), "max:", np.max(mean_d), catalog[np.argmax(mean_d)])
    print("Star number")
    print(catalog[np.argmin(N)], "min:", np.min(N), "max:", np.max(N), catalog[np.argmax(N)])


print("CASE STUDIES")
# for ncat catalog in enumerate(cluster_name_list):
for ncase, case in enumerate(case_study_names):
    OC = star_cluster(case, case_study_dfs[ncase], catalog_mode=False)
    print(OC.name)
    mean_d = np.mean(OC.distance)
    N = OC.Nstars
    print("Distance:", mean_d)
    print("N stars:", N)

