# created 15-3-23
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVR

import numpy as np
import pandas as pd

from Plotting_essentials import CMD_density_design
from sklearn.preprocessing import MinMaxScaler
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

    def __init__(self, name: str, catalog: pd.DataFrame, catalog_mode: bool = True):  # --> works in main

        # Step 1: slashes in clusternames are causing problems (CATALOG III)

        if "/" in name:
            self.name = name.replace("/", "-")
        else:
            self.name = name

        # Step 2: read in cluster data
        self.data = catalog[catalog.Cluster_id == name]
        if catalog_mode:
            self.catalog_id = self.data["catalog"].unique()[0]
        else:
            self.catalog_id = None
        # Step 3: Global cluster properties do not change with changing CMDs
        if 'plx' in self.data.columns:
            self.distance = 1000 / self.data.plx
        else:
            self.distance = None
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

    def create_CMD(self, CMD_params: list = None, return_errors: bool = False, no_errors: bool = False):  # --> works in main

        if not CMD_params:
            self.CMD_specs = dict(axes=["Gmag", "BP-RP"], filters=["Gmag", "BPmag", "RPmag"], short="G_BPRP")
            CMD_params = self.CMD_specs["axes"]
            mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]
        else:
            if len(CMD_params) == 2:
                mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]
                self.CMD_specs = dict(axes=CMD_params, filters=[CMD_params[0], CMD_params[1].split("-")[0] + "mag",
                                                                CMD_params[1].split("-")[1] + "mag"],
                                      short=CMD_params[0].replace("mag", "") + "_" + CMD_params[1].replace("-", ""))
            elif len(CMD_params) == 3:
                mag, p1, p2 = self.data[CMD_params[0]], self.data[CMD_params[1]], self.data[CMD_params[2]]
                cax = p1 - p2
                self.CMD_specs = dict(
                    axes=[CMD_params[0],
                          str(CMD_params[1].replace("mag", "") + "-" + CMD_params[2].replace("mag", ""))],
                    filters=CMD_params, short=CMD_params[0].replace("mag", "") + "_" + CMD_params[1].replace("mag", "")
                                              + CMD_params[2].replace("mag", ""))
            else:
                print("Check CMD params")
                mag = None
                cax = None

        abs_mag = (mag - 5 * np.log10(self.distance) + 5)

        arr = np.stack([cax, abs_mag], axis=1)
        # first remove nans
        nan_idxes = np.isnan(arr).any(axis=1)
        cleaned_arr = arr[~nan_idxes]
        # then sort the array along the yaxis
        sort_idxes = cleaned_arr[:, 1].argsort()
        sorted_arr = cleaned_arr[sort_idxes]

        self.CMD = sorted_arr
        self.N_CMD = len(sorted_arr)

        self.pca = PCA(n_components=2)
        self.PCA_XY = self.pca.fit_transform(sorted_arr)

        # Plotting variables
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)

        # weights
        if not no_errors:
            if not return_errors:
                self.create_weights(sorting_ids=sort_idxes, nan_ids=nan_idxes)
            else:
                errors = self.create_weights(sorting_ids=sort_idxes, nan_ids=nan_idxes, return_deltas=return_errors)
                return errors

    def create_weights(self, sorting_ids: np.ndarray, nan_ids: np.ndarray, cols: list = None,
                       return_deltas: bool = False):  # --> works in main

        min_max_scaler = MinMaxScaler()

        if not cols:
            cols = ["plx", "e_plx"]

        # Work exactly like the Create_CMD function: first remove the nan, then argsort in the same manner
        CMD_errors_nonan = self.errors[~nan_ids]
        CMD_error_values = CMD_errors_nonan.values
        CMD_errors = pd.DataFrame(CMD_error_values[sorting_ids], CMD_errors_nonan.index[sorting_ids],
                                  CMD_errors_nonan.columns)

        CMD_plx_nonan = self.data[cols[0]][~nan_ids].to_numpy()
        CMD_plx = CMD_plx_nonan[sorting_ids]

        delta_m = CMD_errors.filter(regex=self.CMD_specs["filters"][0]).to_numpy().reshape(len(CMD_errors), )
        delta_c1 = CMD_errors.filter(regex=self.CMD_specs["filters"][1]).to_numpy().reshape(len(CMD_errors), )
        delta_c2 = CMD_errors.filter(regex=self.CMD_specs["filters"][2]).to_numpy().reshape(len(CMD_errors), )

        delta_cax = cax_error(delta_c1, delta_c2)
        zero_indices = np.where(delta_cax is False)[0]
        m = min(i for i in delta_cax if i > 0)
        for z_id in zero_indices:
            delta_cax[z_id] = m

        try:
            delta_Mabs = abs_mag_error(CMD_plx, CMD_errors[cols[1]], delta_m).to_numpy()
            weights = (1 / RSS(delta_Mabs, delta_cax)).reshape(len(delta_cax), 1)

        except KeyError:
            print("No plx errors found, using only color axis errors for the weight calculation.")
            weights = (1 / delta_cax).reshape(len(delta_cax), 1)

        if all(weights >= 0):
            self.weights = min_max_scaler.fit_transform(weights).reshape(len(weights), )
        else:
            self.weights = np.ones(len(weights))
            print("No errors found for the CMD data. Setting weight array to unity.")

        if return_deltas:
            try:
                return [delta_c1, delta_c2, delta_m, delta_cax, delta_Mabs]
            except UnboundLocalError:
                return [delta_c1, delta_c2, delta_m, delta_cax]
    @staticmethod
    def gridsearch_and_ranking(X_train: np.ndarray, Y_train: np.ndarray, grid: dict,
                               weight_train: np.ndarray):  # --> works in main

        # 1. create a cross-validation via 5-folds and define the hyperparameter search function
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)
        search = GridSearchCV(estimator=SVR(), param_grid=grid, cv=rkf)

        # 2. fit the defined search function to the training data and return a formatted result pd.DataFrame
        search.fit(X_train, Y_train, sample_weight=weight_train)
        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values(by=["rank_test_score"])
        results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("kernel")
        return results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

    def SVR_read_from_file(self, file: str, catalog_mode: bool = True):  # --> works in main

        # 1. load the file containing all saved hyperparameters as pd.DataFrame
        hp_df = pd.read_csv(file)

        # 2. filter the exact HP needed for 1) a specific cluster and 2) a specific CMD
        if not catalog_mode:
            filtered_values = np.where(
                (hp_df['name'] == self.name) &  # (1
                (hp_df['abs_mag'] == self.CMD_specs['axes'][0]) & (hp_df['cax'] == self.CMD_specs['axes'][1]))[0]  # (2
        # * and 3) catalog
        else:
            filtered_values = np.where(
                (hp_df['name'] == self.name) & (hp_df["catalog_id"] == self.catalog_id) &  # (1 + (3
                (hp_df['abs_mag'] == self.CMD_specs['axes'][0]) & (hp_df['cax'] == self.CMD_specs['axes'][1]))[0]  # (2
            # print(hp_df.loc[filtered_values])
        # 3. the columns with the SVR tuning parameters are grabbed in the form of a dict
        params = hp_df.loc[filtered_values][['C', 'epsilon', 'gamma', 'kernel']].to_dict(orient="records")[0]

        return params

    def SVR_Hyperparameter_tuning(self, input_array: np.ndarray, weight_data: np.ndarray, output_file: str = None,
                                  grid: dict = None, catalog_mode: bool = True):  # --> works in main

        # 1. Split and reshape the input array for the train_test_split() function
        X_data = input_array[:, 0].reshape(len(input_array[:, 0]), 1)
        # * the sample weights are passed with the y values to uphold the right value-weight combinations in the split
        Y_data = np.stack([input_array[:, 1], weight_data], axis=1)

        # 2. Define a training and test set
        X_tr, X_test, Y_tr, Y_test = train_test_split(X_data, Y_data, random_state=13)

        # 3. Scale training set
        X_mean = np.mean(X_tr)
        X_std = np.std(X_tr)
        X_train_scaled = (X_tr - X_mean) / X_std

        # 4. Split the y data and weights again
        Y_flat = Y_tr[:, 0].ravel()
        Y_weights = Y_tr[:, 1].copy(order="C")  # make the weight array C-writeable (Python bug)

        # 4. Create parameter grid (These are the tested grid values from the Working_SVR_isochrones.py file)
        if grid is None:
            evals = np.logspace(-2, -1.5, 20)
            Cvals = np.logspace(-2, 2, 20)
            grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals, epsilon=evals)

        # 5. Call the gridsearch function
        ranking = self.gridsearch_and_ranking(X_train=X_train_scaled, Y_train=Y_flat, grid=grid, weight_train=Y_weights)
        print("...finished tuning")

        # 8. Write the output to the Hyperparameter file:
        # Cluster name and CMD specs MUST be included for unique identification of the correct hyperparameters
        if catalog_mode:
            output_data = {"name": self.name, "abs_mag": self.CMD_specs["axes"][0], "cax": self.CMD_specs["axes"][1],
                           "score": ranking.mean_test_score[0], "std": ranking.std_test_score[0],
                           "catalog_id": self.catalog_id}
        else:
            output_data = {"name": self.name, "abs_mag": self.CMD_specs["axes"][0], "cax": self.CMD_specs["axes"][1],
                           "score": ranking.mean_test_score[0], "std": ranking.std_test_score[0]}
            # new
        if output_file:
            output_data = output_data | ranking.params[0]

            df_row = pd.DataFrame(data=output_data, index=[0])
            df_row.to_csv(output_file, mode="a", header=False)

        else:
            print(ranking.params[0])
            print("score:", ranking.mean_test_score[0])
            print("std:", ranking.std_test_score[0])

        # 8. return params for check
        return ranking.params[0]

    def curve_extraction(self, svr_data: np.ndarray, HP_file: str, svr_predict: np.ndarray = None,
                         svr_weights: np.ndarray = None,
                         grid: dict = None, always_tune: bool = False, catalog_mode: bool = True):  # --> works in main

        # 1. Define the array which is used as base for the prediction of the curve
        # If the svr_data is an array of bootstrapped values, the prediction still needs to be performed on the
        # original X data, otherwise the Confidence borders will not be smooth
        if svr_predict is None:
            svr_predict = svr_data[:, 0].reshape(len(svr_data[:, 0]), 1)
        else:
            svr_predict = svr_predict[:, 0].reshape(len(svr_predict[:, 0]), 1)

        # 2. Define the array that is used for the sample weights if it is not given
        if svr_weights is None:
            try:
                svr_weights = self.weights
            except TypeError:
                print("No weight data found: Weights set to unity")
                svr_weights = np.ones(len(svr_data[:, 1]))

        # 3. Either read in the SVR parameters from the HP file, or determine them by tuning first
        if not always_tune:
            try:
                params = self.SVR_read_from_file(file=HP_file, catalog_mode=catalog_mode)
            except IndexError:
                print("Index error: Running HP tuning for {}...".format(self.name))
                params = self.SVR_Hyperparameter_tuning(input_array=svr_data, weight_data=svr_weights,
                                                        output_file=HP_file,
                                                        grid=grid, catalog_mode=catalog_mode)
        else:
            params = self.SVR_Hyperparameter_tuning(input_array=svr_data, weight_data=svr_weights, output_file=None,
                                                    grid=grid, catalog_mode=catalog_mode)

        # 4. Define the two coordinates for SVR and fit-predict the tuned model to them
        X = svr_data[:, 0].reshape(len(svr_data[:, 0]), 1)
        Y = svr_data[:, 1]
        svr_model = SVR(**params)
        Y_all = svr_model.fit(X, Y, sample_weight=svr_weights).predict(svr_predict)

        # 5. The results are a PCA curve and the corresponding isochrone
        curve = np.stack([svr_predict[:, 0], Y_all], 1)
        rev_transform = self.pca.inverse_transform(curve)

        return curve, rev_transform

    def resample_curves(self, idx: int, output: np.ndarray, sampling_array: np.ndarray = None,
                        sampling_weights: np.ndarray = None, kwargs: dict = None):

        # 0. If no sampling array or weights are given, use the cluster attributes PCA_XY and weights
        if sampling_array is None:
            sampling_array = self.PCA_XY
        if sampling_weights is None:
            sampling_weights = self.weights

        # 1. Stack the XY array and the weights to allow for joint resampling
        XY_weights_stack = np.stack([sampling_array[:, 0], sampling_array[:, 1], sampling_weights], axis=1)
        bs = resample(XY_weights_stack)

        # 2. Split up the resampled array again and calculate the (PCA) isochrone from the resampled data
        bs_XY, bs_weights = bs[:, :2].copy(order="C"), bs[:, 2].copy(order="C")
        curve, isochrone = self.curve_extraction(svr_data=bs_XY, svr_weights=bs_weights, svr_predict=sampling_array,
                                                 **kwargs)

        # 3. Write the result in the provided output file
        output[:, :2, idx] = curve
        output[:, 2:4, idx] = isochrone

    def interval_stats(self, n_resample: int, sampling_array=None, sampling_weights=None, njobs: int = None,
                       kwargs: dict = None):

        if not njobs:
            njobs = 6

        # 1. Create output array where all resampled curves will be stowed
        isochrone_array = np.empty(shape=(len(self.PCA_XY[:, 0]), 4, n_resample))

        # 2. Parallelized generation of the resampled curves
        tic = time.time()
        Parallel(n_jobs=njobs, require="sharedmem")(
            delayed(self.resample_curves)(idx, output=isochrone_array, sampling_array=sampling_array,
                                          sampling_weights=sampling_weights, kwargs=kwargs) for idx in
            range(n_resample))
        toc = time.time()
        print(toc - tic, "s parallel")

        # 3. Create an array holding the stats and walk through all possible x values = svr_predict
        stats_array = np.empty(shape=(len(isochrone_array[:, 0, :]), 3))
        for j, x_i in enumerate(self.PCA_XY[:, 0]):
            PCA_y_vals = []

            # 4. Over all the resampled curves, collect the Y values corresponding to the current X value
            # (there could be multiple Ys for each X due to the bootstrapping)
            for i in range(n_resample):
                PCA_y_vals.extend(isochrone_array[ii, 1, i] for ii in np.where(isochrone_array[:, 0, i] == x_i)[0])

            # 5. calculate the Median and percentiles for each X value from all bootstrapped curves
            PCA_y_median = np.median(PCA_y_vals)
            PCA_y_lower, PCA_y_upper = np.percentile(PCA_y_vals, [5, 95])
            stats_array[j, :] = PCA_y_lower, PCA_y_median, PCA_y_upper

        return stats_array

    def isochrone_and_intervals(self, n_boot: int, data: np.ndarray = None, weights: np.ndarray = None,
                                parallel_jobs: int = None, output_loc: str = None, kwargs: dict = None):

        # 0. Define x_array for the stacking and reverse transformation
        if data is None:
            x_array = self.PCA_XY[:, 0]
        else:
            x_array = data[:, 0]

        # 1. Call the interval_stats() function to get the PCA stats data for the Y variable
        y_array = self.interval_stats(n_resample=n_boot, sampling_array=data, sampling_weights=weights,
                                      njobs=parallel_jobs, kwargs=kwargs)

        sorted_array = np.empty(shape=(len(x_array), 6))
        col_counter = 0

        # 2. stack the X array to each of the Y arrays and transform the coordinate pairs back into CMD curves
        for col in range(3):
            stack = np.stack([x_array, y_array[:, col]], axis=1)
            rev_stack = self.pca.inverse_transform(stack)
            sorted_stack = rev_stack[rev_stack[:, 1].argsort()]
            sorted_array[:, col_counter:col_counter + 2] = sorted_stack
            col_counter += 2

        # 3. Create a pd.Dataframe that is returned and can also be saved
        new_cols = ["l_x", "l_y", "m_x", "m_y", "u_x", "u_y"]
        sorted_df = pd.DataFrame(data=sorted_array, columns=new_cols)

        if output_loc:
            output_file = "{0}_{1}_nboot_{2}_cat_{3}.csv".format(self.name, self.CMD_specs["short"], n_boot,
                                                                 self.catalog_id)
            sorted_df.to_csv(output_loc + output_file, mode="w", header=True)

        return sorted_df

# WORKS and is UPTODATE 3-4-23
# ============================

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from pre_processing import cluster_df_list, cluster_name_list
#
#     CI_clusters, CI_df = cluster_name_list[0], cluster_df_list[0]
#
#     HPfile = "data/Hyperparameters/CatalogI.csv"
#     try:
#         pd.read_csv(HPfile)
#     except FileNotFoundError:
#         with open(HPfile, "w") as f:
#             f.write("id,name,abs_mag,cax,score,std,C,epsilon,gamma,kernel\n")
#
#     HP_kwargs = {"HP_file": HPfile, "grid": None}
#
#     for cluster in CI_clusters[:3]:
#         OC = star_cluster(cluster, CI_df)
#         OC.create_CMD()
#         print(OC.name)
#
#         try:
#             HP_params = OC.SVR_read_from_file(HPfile)
#         except IndexError:
#             c, i = OC.curve_extraction(OC.PCA_XY, **HP_kwargs)
#
#         # manual bootstrapping, not using master function
#         nboot = 100
#         isochrone_arr = np.empty(shape=(len(OC.PCA_XY[:, 0]), 4, nboot))
#         Parallel(n_jobs=6, require="sharedmem")(
#             delayed(OC.resample_curves)(idx=idx, output=isochrone_arr, kwargs=HP_kwargs) for
#             idx in range(nboot))
#
#         f = CMD_density_design(OC.CMD, cluster_obj=OC)
#         for i in range(nboot):
#             plt.plot(isochrone_arr[:, 2, i], isochrone_arr[:, 3, i], color="orange")
#         f.show()
#
#         g = CMD_density_design(OC.PCA_XY, cluster_obj=OC)
#         for i in range(nboot):
#             plt.plot(isochrone_arr[:, 0, i], isochrone_arr[:, 1, i], color="orange")
#         g.show()
#
#         fes = OC.isochrone_and_intervals(n_boot=nboot, data=OC.PCA_XY, weights=OC.weights,
#                                          parallel_jobs=6, kwargs=HP_kwargs)
#
#         fig2 = CMD_density_design(OC.CMD, cluster_obj=OC)
#
#         plt.plot(fes["l_x"], fes["l_y"], color="grey")
#         plt.plot(fes["m_x"], fes["m_y"], color="red")
#         plt.plot(fes["u_x"], fes["u_y"], color="grey")
#
#         plt.show()
