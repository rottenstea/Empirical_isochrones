import seaborn as sns
import matplotlib.pyplot as plt

from EmpiricalArchive.My_tools import my_utility
from EmpiricalArchive.Extraction.Classfile import *
from sklearn.model_selection import RepeatedKFold

# 0.1 Set the correct output paths
output_path = my_utility.set_output_path()
results_path = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/"
isochrone_path = "../data/Isochrones/Empirical/"
# 0.2 HP file check
HP_file = "../data/Hyperparameters/Carina_TWA.csv"
my_utility.setup_HP(HP_file)

# 0.4 Set the kwargs for the parameter grid and HP file and plot specs
evals = np.logspace(-1., -1.5, 100)
Cvals = np.logspace(-1., 1, 100)
grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals, epsilon=evals)
kwargs = dict(grid=grid, HP_file=HP_file, rkf_func=RepeatedKFold(n_splits=3, n_repeats=1, random_state=13))

# 0.5 Standard plot settings
sns.set_style("darkgrid")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 18

save_plot = False

TW_df = pd.read_csv("../data/Cluster_data/TW_Hydrae/TWA_single_p68_w_errors_DR32016.csv")
TW_df["Cluster_id"] = "TW_Hydrae"

Carina_all_df = pd.read_csv("../data/Cluster_data/Carina/Carina_sources.csv")
Carina_df = Carina_all_df[Carina_all_df["bonafide_UVES"] == True]
Carina_df.loc[:, "Cluster_id"] = "Carina"
Carina_df.loc[:, "BP-G"] = Carina_df["BPmag"] - Carina_df["Gmag"]
Carina_df.loc[:, "G-RP"] = Carina_df["Gmag"] - Carina_df["RPmag"]

Carina_df.rename(columns={"parallax": "Plx", "parallax_error": "e_Plx"}, inplace=True)

dfs = [Carina_df, TW_df]

for df in dfs[:]:
    df_focus = df.rename(columns={'Plx': 'plx', 'e_Plx': 'e_plx'})

    for CMD_axes in ["BP-RP", "BP-G", "G-RP"]:

        # 1. Create a class object for each cluster
        OC = star_cluster(np.unique(df_focus["Cluster_id"])[0], df_focus, dataset_id="ad")

        # 2. Create the CMD that should be used for the isochrone extraction
        OC.create_CMD(CMD_params=["Gmag", CMD_axes])

        # 3. Do some initial HP tuning if necessary
        try:
            params = OC.SVR_read_from_file(HP_file)
        except IndexError:
            print(f"No Hyperparameters were found for {OC.name}.")
            curve, isochrone = OC.curve_extraction(svr_data=OC.PCA_XY, svr_weights=OC.weights,
                                                   svr_predict=OC.PCA_XY[:, 0], **kwargs)

        # 4. Create the robust isochrone and uncertainty border from bootstrapped curves
        n_boot = 1_000
        result_df = OC.isochrone_and_intervals(n_boot=n_boot, kwargs=kwargs, output_loc=isochrone_path)

        # 5. Plot the result
        fig = CMD_density_design(OC.CMD, cluster_obj=OC)

        plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
        plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
        plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

        fig.savefig(output_path + f'{OC.name}_{CMD_axes}.png', dpi=300)
        plt.show()
