# created 15-3-23
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVR

import numpy as np
import pandas as pd


from Plotting_essentials import CMD_density_design
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from joblib import Parallel, delayed
import time


def abs_mag_error(w, delta_w, delta_m):
    return np.sqrt((5 / (np.log(10) * w) * delta_w) ** 2 + delta_m ** 2)


def cax_error(delta_c1, delta_c2):
    return np.sqrt(delta_c1 ** 2 + delta_c2 ** 2)


def RSS(e1, e2):
    return np.sqrt(e1 ** 2 + e2 ** 2)


class star_cluster(object):

    def __init__(self, name: str, catalog: pd.DataFrame):

        # Step 1: slashes in clusternames are causing problems (CATALOG III)

        if "/" in name:
            self.name = name.replace("/", "|")
        else:
            self.name = name

        # Step 2: read in cluster data
        self.data = catalog[catalog.Cluster_id == name]

        # Step 3: Global cluster properties do not change with changing CMDs
        self.Catalog = None
        self.distance = 1000 / self.data.plx
        self.Nstars = len(self.data)

        age_cols = [col for col in self.data.columns if 'age' in col]
        self.ages = self.data[age_cols].drop_duplicates()

        av_cols = [col for col in self.data.columns if 'av' in col]
        self.avs = self.data[av_cols].drop_duplicates()

        error_cols = [col for col in self.data.columns if str(col).startswith("e_")]
        self.errors = self.data[error_cols]

        # Step 4: Initialize all kinds of variables (some might not even be needed anymore, so REVISIT LATER)

        # CMD variables
        self.CMD = None
        self.CMD_specs = None
        self.N_CMD = None
        self.density_x, self.density_y, self.kwargs_CMD = None, None, None

        self.weights = None

        # PCA variables
        self.PCA_XY = None
        self.pca = None

    def create_CMD(self, sort_idx: int = 1, CMD_params: list = None):

        if not CMD_params:
            self.CMD_specs = dict(axes=["Gmag", "BP-RP"], filters=["Gmag", "BPmag", "RPmag"])
            CMD_params = self.CMD_specs["axes"]
            mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]
        else:
            if len(CMD_params) == 2:
                mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]
                self.CMD_specs = dict(axes=CMD_params, filters=[CMD_params[0], CMD_params[1].split("-")[0] + "mag",
                                                                CMD_params[1].split("-")[1] + "mag"])
            elif len(CMD_params) == 3:
                mag, p1, p2 = self.data[CMD_params[0]], self.data[CMD_params[1]], self.data[CMD_params[2]]
                cax = p1 - p2
                self.CMD_specs = dict(
                    axes=[CMD_params[0],
                          str(CMD_params[1].replace("mag", "") + "-" + CMD_params[2].replace("mag", ""))],
                    filters=CMD_params)
            else:
                print("Check CMD params")
                mag = None
                cax = None

        abs_mag = (mag - 5 * np.log10(self.distance) + 5)

        arr = np.stack([cax, abs_mag], axis=1)
        cleaned_arr = arr[~np.isnan(arr).any(axis=1), :]
        nan_idxes = np.isnan(arr).any(axis=1)
        sorted_arr = cleaned_arr[cleaned_arr[:, sort_idx].argsort()]

        self.CMD = sorted_arr
        self.N_CMD = len(sorted_arr)

        self.pca = PCA(n_components=2)
        self.PCA_XY = self.pca.fit_transform(sorted_arr)

        # weights
        self.create_weights(contaminated_idxes=nan_idxes)

        # Plotting variables
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)

    def create_weights(self, contaminated_idxes, cols: list = None):

        if not cols:
            cols = ["plx", "e_plx"]

        CMD_errors = self.errors[~contaminated_idxes]
        CMD_plx = self.data[cols[0]][~contaminated_idxes]

        delta_m = CMD_errors.filter(regex=self.CMD_specs["filters"][0]).to_numpy().reshape(len(CMD_errors), )
        delta_c1 = CMD_errors.filter(regex=self.CMD_specs["filters"][1]).to_numpy().reshape(len(CMD_errors), )
        delta_c2 = CMD_errors.filter(regex=self.CMD_specs["filters"][2]).to_numpy().reshape(len(CMD_errors), )

        try:
            delta_Mabs = abs_mag_error(CMD_plx, CMD_errors[cols[1]], delta_m).to_numpy()
            delta_cax = cax_error(delta_c1, delta_c2)

            self.weights = 1 / RSS(delta_Mabs, delta_cax)

        except KeyError:
            print("keyerror")

            self.weights = 1 / cax_error(delta_c1, delta_c2)

    @staticmethod
    def gridsearch_and_ranking(X_train, Y_train, rkf, pg, weight):


        search = GridSearchCV(estimator=SVR(), param_grid=pg, cv=rkf)

        search.fit(X_train, Y_train, sample_weight=weight)

        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values(by=["rank_test_score"])
        results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("kernel")
        return results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

    def SVR_read_from_file(self, HP_file):
        hp_df = pd.read_csv(HP_file)
        #print("hp_df:", hp_df)

        filtered_values = np.where(
            (hp_df['name'] == self.name) & (hp_df['abs_mag'] == self.CMD_specs['axes'][0]) &
            (hp_df['cax'] == self.CMD_specs['axes'][1]))[0]

        #print("filtered vals:",filtered_values)

        params = hp_df.loc[filtered_values][['C', 'epsilon', 'gamma', 'kernel']].to_dict(orient="records")[0]
        #print("params:",params)
        return params

    def SVR_Hyperparameter_tuning(self, array, weight_data, grid_dict: dict = None, HP_file=None):

        X_data = array[:, 0].reshape(len(array[:, 0]), 1)
        Y_data = np.stack([array[:, 1], weight_data], axis=1)

        # 2. Split X and Y into training and test set
        X_tr, X_test, Y_tr, Y_test = train_test_split(X_data, Y_data, random_state=13)

        # 3. Scale training and test set wrt. to the training set
        X_mean = np.mean(X_tr)
        X_std = np.std(X_tr)
        X_train_scaled = (X_tr - X_mean) / X_std
        # X_test_scaled = (X_test - X_mean) / X_std
        # X_scaled = np.array((X_data - X_mean) / X_std)
        Y_flat = Y_tr[:, 0].ravel()
        Y_err = Y_tr[:, 1].copy(order="C")

        # 4. Create parameter grid
        if grid_dict is None:
            kernels = ["rbf"]
            C_range = np.logspace(-1, 2, 20)
            epsilon_range = np.logspace(-6, 1, 20)

            grid = [
                dict(kernel=kernels, gamma=["auto"], C=C_range,
                     epsilon=epsilon_range), ]

        else:
            grid = [grid_dict, ]

        #5. define crossvalidation method
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)

        # 5. call the gridsearch function
        ranking = self.gridsearch_and_ranking(X_train=X_train_scaled, Y_train=Y_flat, rkf = rkf, pg=grid, weight=Y_err)
        print("fin")

        # 7. Write output to file
        if HP_file:
            output_data = {"name": self.name, "abs_mag": self.CMD_specs["axes"][0], "cax": self.CMD_specs["axes"][1],
                           "score": ranking.mean_test_score[0], "std": ranking.std_test_score[0]}
            output_data = output_data | ranking.params[0]

            df_row = pd.DataFrame(data=output_data, index=[0])
            df_row.to_csv(HP_file, mode="a", header=False)

        else:
            print(ranking.params[0])
            print("score:", ranking.mean_test_score[0])
            print("std:", ranking.std_test_score[0])

        # 8. return params for check
        return ranking.params[0]

    def curve_extraction(self, array, HP_file, grid=None):

        if self.weights is None:
            weight_data = np.ones(shape=(len(array[:, 1])))
        else:
            weight_data = self.weights

        X = array[:, 0].reshape(len(array[:, 0]), 1)
        Y = array[:, 1]

        try:
            params = self.SVR_read_from_file(HP_file=HP_file)
        except IndexError:
            print("Index error")
            params = self.SVR_Hyperparameter_tuning(array, weight_data, grid, HP_file=HP_file)

        svr = SVR(**params)

        Y_pred = svr.fit(X, Y, sample_weight=weight_data).predict(X)
        curve = np.stack([X[:, 0], Y_pred], 1)
        curve = curve[curve[:, 0].argsort()]

        return curve

    def resample_curves(self, data, output, idx, HP_file, grid=None):

        bs = resample(data, n_samples=(len(data[:, 0])))
        curve = self.curve_extraction(bs, HP_file, grid)
        isochrone = self.pca.inverse_transform(curve)

        output[:, :2, idx] = curve
        output[:, 2:4, idx] = isochrone

    def interval_stats(self, array, n_res, HP_file, grid=None):

        isochrone_array = np.empty(shape=(len(self.PCA_XY[:, 0]), 4, n_res))

        tic = time.time()
        Parallel(n_jobs=6, require="sharedmem")(
            delayed(self.resample_curves)(array, isochrone_array, idx, HP_file, grid) for idx in range(n_res))
        toc = time.time()
        print(toc - tic, "s parallel")

        # SERIAL
        # for i in range(n_res):
        #    self.resample_curves(array, isochrone_array, i, HP_file, grid)

        stats_array = np.empty(shape=(len(isochrone_array[:, 0, :]), 3))

        # n_boot
        n_iter = len(isochrone_array[0, 0, :])
        # walk through all possible x values on the color axis of the CMD
        for j, x_i in enumerate(self.PCA_XY[:, 0]):

            PCA_y_vals = []
            for i in range(n_iter):
                PCA_y_vals.extend(isochrone_array[ii, 1, i] for ii in np.where(isochrone_array[:, 0, i] == x_i)[0])

            PCA_y_median = np.median(PCA_y_vals)
            PCA_y_lower, PCA_y_upper = np.percentile(PCA_y_vals, [5, 95])
            stats_array[j, :] = PCA_y_lower, PCA_y_median, PCA_y_upper

        return self.PCA_XY[:, 0], stats_array

    def isochrone_and_intervals(self, array, n_res, index: int = 1, HP_file=None, grid=None, output_loc=None):

        if not output_loc:
            output_loc = "data/isochrones/{}.csv".format(self.name)

        x_array, y_array = self.interval_stats(array, n_res, HP_file, grid)

        sorted_array = np.empty(shape=(len(x_array), 6))
        col_number = len(y_array[0, :])
        col_counter = 0

        for col in range(col_number):
            stack = np.stack([x_array, y_array[:, col]], axis=1)
            rev_stack = self.pca.inverse_transform(stack)
            sorted_stack = rev_stack[rev_stack[:, index].argsort()]
            sorted_array[:, col_counter:col_counter + 2] = sorted_stack
            col_counter += 2

        new_cols = ["l_x", "l_y", "m_x", "m_y", "u_x", "u_y"]
        sorted_df = pd.DataFrame(data=sorted_array, columns=new_cols)
        sorted_df.to_csv(output_loc, mode="w", header=True)
        return sorted_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pre_processing import create_df

    CI_raw = "data/Cluster_data/all_ages/CatalogI_BCD_ages.csv"

    CI_cols = ["Cluster", "Plx", "e_Plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
               "G-RP",
               "logA_B", "AV_B", "AgeNN_CG", "AVNN_CG", "logage_D", "Av_D",
               "RUWE", "Proba"]

    CI_names = ["Cluster_id", "plx", "e_plx", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "BP-RP", "BP-G",
                "G-RP",
                "age_B", "av_B", "age_C", "av_C", "age_D", "av_D", "ruwe", "probability"]

    q_filter = {"parameter": ["ruwe", "plx", "probability"], "limit": ["upper", "lower", "lower"],
                "value": [1.4, 0, 0.49]}

    CI_clusters, CI_df = create_df(CI_raw, CI_cols, CI_names, q_filter)

    HP_file = "data/Hyperparameters/CatalogI.csv"
    try:
        pd.read_csv(HP_file)
    except FileNotFoundError:
        with open(HP_file, "w") as f:
            f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")

    kwargs = {"HP_file": HP_file, "grid": None}

    for cluster in CI_clusters[:3]:
        OC = star_cluster(cluster, CI_df)
        OC.create_CMD()
        print(OC.name)

        OC.weights = np.ones(shape=(len(OC.PCA_XY[:, 1])))
        curv = OC.curve_extraction(OC.PCA_XY, **kwargs)
        isoc = OC.pca.inverse_transform(curv)
        n_boot = 200


        isochrone_array = np.empty(shape=(len(OC.PCA_XY[:, 0]), 4, n_boot))
        Parallel(n_jobs=6, require="sharedmem")(
            delayed(OC.resample_curves)(OC.PCA_XY, isochrone_array, idx, HP_file) for idx in range(n_boot))


        f = CMD_density_design(OC.CMD, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_array[:, 2, i], isochrone_array[:, 3, i], color="orange")
        f.show()

        g = CMD_density_design(OC.PCA_XY, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_array[:, 0, i], isochrone_array[:, 1, i], color="orange")
        g.show()

        #fes = OC.isochrone_and_intervals(OC.PCA_XY, n_boot, **kwargs)

        #fig2 = CMD_density_design(OC.CMD, cluster_obj=OC)


        #plt.plot(fes["l_x"], fes["l_y"], color="grey")
        #plt.plot(fes["m_x"], fes["m_y"], color="red")
        #plt.plot(fes["u_x"], fes["u_y"], color="grey")
        #plt.plot(isoc[:, 0], isoc[:, 1], color="lime")

        #plt.show()
