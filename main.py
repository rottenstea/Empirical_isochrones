import os
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd

from Classfile import *
from pre_processing import create_df

main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"
subdir = date.today()
output_path = os.path.join(main, str(subdir))
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
output_path = output_path + "/"

# HP check
HP_file = "data/Hyperparameters/CatalogIII.csv"
try:
    pd.read_csv(HP_file)
except FileNotFoundError:
    with open(HP_file, "w") as f:
        f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

CIII_raw = "data/Cluster_data/all_ages/CatalogIII_DR3_Seb_ages.csv"

CIII_cols = ["cluster_name", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
             "BP-G", "G-RP",
             "logage_lts", "logage_tdist",
             "ruwe", "fidelity_v2", "stability", "G_err", "G_BPerr", "G_RPerr"]

CIII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
              "BP-G", "G-RP",
              "age_lts", "age_tdist", "ruwe", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"]

q_filter = {"parameter": ["ruwe", "plx", "fidelity", "stability", "G_err", "G_BPerr", "G_RPerr"],
            "limit": ["upper", "lower", "lower", "lower", "upper", "upper", "upper"], "value": [1.4, 0, 0.5, 25, 0.007, 0.15, 0.03]}


CIII_clusters, CIII_df = create_df(CIII_raw, CIII_cols, CIII_names, q_filter)

kernels = ["rbf"]
C_range = np.logspace(-1, 2, 10)
gamma_range = np.logspace(-5, -1, 8)
epsilon_range = np.logspace(-6, 1, 10)

# svr_grid_rbf = dict(kernel=kernels, gamma=gamma_range, C=C_range)

grid = dict(kernel=kernels, gamma=["auto"], C=C_range,
         epsilon=epsilon_range)


kwargs = {"HP_file": HP_file, "grid": None}


for cluster in CIII_clusters[:]:
    OC = star_cluster(cluster, CIII_df)

    if OC.Nstars >= 100:
        OC.create_CMD()
        #f= CMD_density_design(OC.CMD, cluster_obj=OC)
        #f.show()

    
        print(OC.name, min(OC.data["stability"]),max(OC.data["stability"]))

        OC.weights = OC.data["stability"] * OC.weights
        #print(min(OC.weights))

        curv = OC.curve_extraction(OC.PCA_XY,**kwargs)
        iso = OC.pca.inverse_transform(curv)
        #OC.SVR_read_from_file("data/Hyperparameters/CatalogI.csv")
        #OC.SVR_Hyperparameter_tuning(OC.PCA_XY,OC.weights,output_file="data/Hyperparameters/CatalogI.csv")
        #OC.SVR_read_from_file("data/Hyperparameters/CatalogI.csv")

        #n_boot = 1000
        #
        #fes = OC.isochrone_and_intervals(OC.PCA_XY, n_boot, **kwargs)
        #
        kr = int(0.04 * OC.Nstars)
        fig2 = CMD_density_design(OC.CMD, cluster_obj=OC)

        #plt.plot(fes["l_x"], fes["l_y"], color="grey")
        #plt.plot(fes["m_x"], fes["m_y"], color="orange")
        #plt.plot(fes["u_x"], fes["u_y"], color="grey")
        plt.plot(iso[kr:,0], iso[kr:,1], color= "red", alpha = 0.5)

        plt.show()

        #fig2.savefig(output_path + "{0}_n{1}_final.png".format(OC.name,n_boot), dpi=500)
