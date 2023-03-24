from sklearn.utils import resample
from Classfile import *
import os


def Isochrone_extraction(cluster, array, weights, file="data/Hyperparameters/CatalogII.csv", grid=None):
    # pca = PCA(n_components=2)
    # pca_data = pca.fit_transform(array)

    X = array[:, 0].reshape(len(array[:, 0]), 1)
    Y = array[:, 1]

    try:
        params = cluster.SVR_read_from_file(HP_file=file)
    except FileNotFoundError:

        with open(file, "w") as f:
            f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

        if not grid:
            grid = dict(kernel=["rbf"], gamma=["scale"], C=np.logspace(-2, 2, 5),
                        epsilon=np.logspace(-4, -2, 5))

        params = cluster.SVR_Hyperparameter_tuning(array, grid, HP_file=file)

    svr = SVR(**params)

    Y_pred = svr.fit(X, Y, sample_weight=weights).predict(X)
    curve = np.stack([X[:, 0], Y_pred], 1)
    curve = curve[curve[:, 0].argsort()]
    # isochrone = pca.inverse_transform(PCA_curve)

    return curve


# currently obsolete
def Isochrone_collection(cluster, weights, n_boot, file="data/Hyperparameters/CatalogII.csv", grid=None):
    isochrone_store = np.empty(shape=(len(cluster.CMD[:, 0]), 4, n_boot))

    for i in range(n_boot):
        print("i=", i + 1)
        bs = resample(cluster.PCA_XY, n_samples=(len(cluster.PCA_XY[:, 0])))

        curve = Isochrone_extraction(cluster, bs, weights, file, grid)
        isochrone = cluster.pca.inverse_transform(curve)
        isochrone_store[:, :2, i] = curve
        isochrone_store[:, 2:4, i] = isochrone

    return isochrone_store


# for parallel computing
def resample_recal(cluster, data, weights, output, idx, file=None, grid=None):
    bs = resample(data, n_samples=(len(data[:, 0])))

    if file or grid:
        curve = Isochrone_extraction(cluster, bs, weights, file, grid)
    else:
        curve = Isochrone_extraction(cluster, bs, weights)

    isochrone = cluster.pca.inverse_transform(curve)

    output[:, :2, idx] = curve
    output[:, 2:4, idx] = isochrone



def interval_stats(isochrone_array, cluster):
    stats_array = np.empty(shape=(len(isochrone_array[:, 0, :]), 3))

    # n_boot
    n_iter = len(isochrone_array[0, 0, :])
    # walk through all possible x values on the color axis of the CMD
    for j, x_i in enumerate(cluster.PCA_XY[:, 0]):

        PCA_y_vals = []
        for i in range(n_iter):
            PCA_y_vals.extend(isochrone_array[ii, 1, i] for ii in np.where(isochrone_array[:, 0, i] == x_i)[0])

        PCA_y_median = np.median(PCA_y_vals)
        PCA_y_lower, PCA_y_upper = np.percentile(PCA_y_vals, [5, 95])
        stats_array[j, :] = PCA_y_lower, PCA_y_median, PCA_y_upper

    return cluster.PCA_XY[:,0], stats_array

def isochrone_and_intervals(isochrone_array, cluster, index: int = 1):

    x_array, y_array = interval_stats(isochrone_array, cluster)

    sorted_array = np.empty(shape=(len(x_array),6))
    col_number = len(y_array[0,:])
    col_counter = 0

    for col in range(col_number):
        stack = np.stack([x_array, y_array[:,col]], axis=1)
        rev_stack = cluster.pca.inverse_transform(stack)
        sorted_stack = rev_stack[rev_stack[:, index].argsort()]
        sorted_array[:,col_counter:col_counter+2] = sorted_stack
        col_counter += 2

    return sorted_array


if __name__ == "__main__":

    from pre_processing import create_df

    CII_raw = "data/Cluster_data/all_ages/CatalogII_BCD_ages.csv"

    CII_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
                "G-RP",
                "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
                "RUWE"]

    CII_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP",
                 "BP-G", "G-RP",
                 "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe"]

    q_filter = {"parameter": ["ruwe", "plx"], "limit": ["upper", "lower"], "value": [1.4, 0]}

    CII_clusters, CII_df = create_df(CII_raw, CII_cols, CII_names, q_filter)

    for cluster in CII_clusters[2:3]:
        OC = star_cluster(cluster, CII_df)
        OC.create_CMD()
        weights = OC.create_weights()

        n_boot = 20
        isochrone_store = np.empty(shape=(len(OC.CMD[:, 0]), 4, n_boot))

        for i in range(n_boot):
            resample_recal(OC, OC.PCA_XY, weights, isochrone_store, i)

        #f = CMD_density_design(OC.CMD, cluster_obj=OC)
        #for i in range(n_boot):
        #    plt.plot(isochrone_store[:, 2, i], isochrone_store[:, 3, i], color="orange")
        #f.show()

        #g = CMD_density_design(OC.PCA_XY, cluster_obj=OC)
        #for i in range(n_boot):
        #   plt.plot(isochrone_store[:, 0, i], isochrone_store[:, 1, i], color="orange")
        #g.show()


        fes = isochrone_and_intervals(isochrone_store, OC)

        #print(res.head())


        fig2 = CMD_density_design(OC.CMD, cluster_obj=OC)

        plt.plot(fes[:, 0], fes[:, 1], color="grey")
        plt.plot(fes[:, 2], fes[:, 3], color="red")
        plt.plot(fes[:, 4], fes[:, 5], color="grey")

        plt.show()

