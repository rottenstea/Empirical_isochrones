import numpy as np
import pandas as pd

import time
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.utils import resample

from EmpiricalArchive.My_tools.plotting_essentials import CMD_density_design


def abs_mag_error(w: float, delta_w: float, delta_m: float):
    """
    Root sum of squares for the absolute magnitude. The first term is the derivation of the distance term in the
    distance modulus.

    :param w: Parallax value
    :param delta_w: Parallax error
    :param delta_m: Apparent magnitude error
    :return: float
    """
    return np.sqrt((5 / (np.log(10) * w) * delta_w) ** 2 + delta_m ** 2)


def RSS(e1: float, e2: float):
    """
    General root sum of squares formula.
    :param e1: Error of variable 1
    :param e2: Error of variable 2
    :return: float
    """
    return np.sqrt(e1 ** 2 + e2 ** 2)


class star_cluster(object):

    def __init__(self, name: str, catalog: pd.DataFrame, dataset_id=None):
        """
        Initiation function of the cluster object. Takes a minimum amount of input parameters, namely only the cluster
        identifier for extracting the right part from the big dataframe and the dataframe itself.

        :param name: Cluster name or identifier. Should not include escape characters
        :param catalog: Input dataframe with standardized columns (from pre-processing).
        :param dataset_id: Unique identifier of the dataset, as more than one dataset may exist for one cluster.
        :return: object
        """

        # Optional: slashes in clusternames are causing problems (CATALOG III)
        # Could be generalized later if exception comes up
        if "/" in name:
            self.name = name.replace("/", "-")
        else:
            self.name = name

        # Read in cluster data
        self.data = catalog[catalog.Cluster_id == name]

        # Catalog identifier is needed as multiple datasets may exist for the same clusters.
        # It tags the hyperparameter file.
        if dataset_id is not None:
            self.dataset_id = dataset_id
        else:
            try:
                if np.isnan(self.data["catalog"].unique()[0]):
                    self.dataset_id = "dataset_1"
                    print(f"No unique dataset_id was set for this dataset. "
                          f"Flag dataset_id was set to {self.dataset_id}.")
                else:
                    self.dataset_id = self.data["catalog"].unique()[0]
            except KeyError:
                print(f"No catalog column specifying the unique dataset_id was found. "
                      f"Flag dataset_id was set to {self.dataset_id}.")
                self.dataset_id = "dataset_1"

        # Set global cluster properties
        if 'plx' in self.data.columns:
            self.distance = 1000 / self.data.plx
        else:
            self.distance = None
        self.Nstars = len(self.data)

        # Optional: Global properties from literature
        # Could also be more generalized in the future
        age_cols = [col for col in self.data.columns if 'age' in col]
        self.ages = self.data[age_cols].drop_duplicates()

        av_cols = [col for col in self.data.columns if 'av' in col]
        self.avs = self.data[av_cols].drop_duplicates()

        # Gaia DR3 AG
        if 'AG' in self.data.columns:
            self.A_G = self.data["AG"]

        error_cols = [col for col in self.data.columns if str(col).startswith("e_")]
        self.errors = self.data[error_cols]

        # Initialize CMD-specific variables
        self.CMD = None
        self.CMD_specs = None  # dictionary: 1) axis labels 2) CMD filter list 3) file suffix for the chosen CMD
        self.N_CMD = None  # number of sources in the CMD
        self.density_x, self.density_y, self.kwargs_CMD = None, None, None  # variables for the density scatterplot
        self.weights = None  # weights for SVR
        self.PCA_XY = None  # PCA-transformed data
        self.pca = None  # PCA instance

    def create_CMD(self, CMD_params: list = None, return_errors: bool = False, no_errors: bool = False):
        """
        Create cluster CMD and calculate 1D errors for each datapoint in the scatterplot.

        :param CMD_params: Variables spanning the CMD. Can be either three filters or one filter and one color index. Y
                           mag before cax in the list.
        :param return_errors: If set to true, a list of all calculated errors (raw and CMD errors) is returned.
        :param no_errors: If set to true, the create_weights function will not be engaged and all weights will be set
                          to one.
        :return: Optional: Error list
        """

        # Default CMD parameters: abs G vs. BP-RP
        if not CMD_params:

            # write the full dictionary
            self.CMD_specs = dict(axes=["Gmag", "BP-RP"], filters=["Gmag", "BPmag", "RPmag"], short="G_BPRP")

            # this works only if a column with a color axis (index) already exists
            CMD_params = self.CMD_specs["axes"]
            mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]

        else:
            # this works only if a column with a color axis (index) already exists
            if len(CMD_params) == 2:
                mag, cax = self.data[CMD_params[0]], self.data[CMD_params[1]]
                self.CMD_specs = dict(axes=CMD_params, filters=[CMD_params[0], CMD_params[1].split("-")[0] + "mag",
                                                                CMD_params[1].split("-")[1] + "mag"],
                                      short=CMD_params[0].replace("mag", "") + "_" + CMD_params[1].replace("-", ""))

            # this always works
            elif len(CMD_params) == 3:
                mag, p1, p2 = self.data[CMD_params[0]], self.data[CMD_params[1]], self.data[CMD_params[2]]
                cax = p1 - p2
                self.CMD_specs = dict(
                    axes=[CMD_params[0],
                          str(CMD_params[1].replace("mag", "-") + CMD_params[2].replace("mag", ""))],
                    filters=CMD_params,
                    short=CMD_params[0].replace("mag", "_") + CMD_params[1].replace("mag", "") + CMD_params[2].
                    replace("mag", ""))
            else:
                print("Check CMD parameters.")
                mag = None
                cax = None

        # compute absolute magnitude and stack with color index
        abs_mag = (mag - 5 * np.log10(self.distance) + 5)
        arr = np.stack([cax, abs_mag], axis=1)

        # remove nans
        nan_idxes = np.isnan(arr).any(axis=1)
        cleaned_arr = arr[~nan_idxes]

        # sort the cleaned array along the yaxis
        sort_idxes = cleaned_arr[:, 1].argsort()
        sorted_arr = cleaned_arr[sort_idxes]

        # set the star_cluster attributes
        self.CMD = sorted_arr
        self.N_CMD = len(sorted_arr)
        self.pca = PCA(n_components=2)
        self.PCA_XY = self.pca.fit_transform(sorted_arr)
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)

        # compute weights
        if not no_errors:
            if not return_errors:
                self.create_weights(sorting_ids=sort_idxes, nan_ids=nan_idxes)
            else:
                errors = self.create_weights(sorting_ids=sort_idxes, nan_ids=nan_idxes, return_deltas=return_errors)
                return errors
        else:
            print("The 'no_errors' flag is activated. All weights for the SVR will be set to one.")
            self.weights = np.ones(self.N_CMD)

    def create_CMD_quick_n_dirty(self, CMD_params, no_errors: bool = True):

        self.CMD_specs = dict(axes=CMD_params, filters=[CMD_params[0], CMD_params[1].split("-")[0] + "mag",
                                                        CMD_params[1].split("-")[1] + "mag"],
                              short=CMD_params[0].replace("mag", "") + "_" + CMD_params[1].replace("-", ""))

        # remove nans
        #abs_mag = self.data[CMD_params[1]]
        #cax = self.data[CMD_params[0]]

        arr = self.data[[CMD_params[1], CMD_params[0]]].to_numpy()

        nan_idxes = np.isnan(arr).any(axis=1)
        cleaned_arr = arr[~nan_idxes]

        # sort the cleaned array along the yaxis
        sort_idxes = cleaned_arr[:, 1].argsort()
        sorted_arr = cleaned_arr[sort_idxes]

        # set the star_cluster attributes
        self.CMD = sorted_arr
        self.N_CMD = len(sorted_arr)
        self.pca = PCA(n_components=2)
        self.PCA_XY = self.pca.fit_transform(sorted_arr)
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)

        # compute weights
        if not no_errors:
            self.create_weights(sorting_ids=sort_idxes, nan_ids=nan_idxes)
        else:
            print("The 'no_errors' flag is activated. All weights for the SVR will be set to one.")
            self.weights = np.ones(self.N_CMD)

    def create_weights(self, sorting_ids: np.ndarray, nan_ids: np.ndarray, plx_or_d_cols: list = None,
                       return_deltas: bool = False):
        """
        Calculate a one-dimensional, scalar weight value for each datapoint in the CMD that will be considered in SVR.

        :param sorting_ids: Sort the error columns of the raw data in the same manner as was done for the CMD.
        :param nan_ids: Clean the errors in all columns if a NaN was encountered in one of the CMD columns.
        :param plx_or_d_cols: List of column names for the distance / plx values and errors.
        :param return_deltas:
        :return:
        """
        min_max_scaler = MinMaxScaler()

        if not plx_or_d_cols:
            plx_or_d_cols = ["plx", "e_plx"]

        # Exactly like the Create_CMD function: First remove the nan, then argsort in the same manner
        CMD_errors_nonan = self.errors[~nan_ids]
        CMD_error_values = CMD_errors_nonan.values
        CMD_errors = pd.DataFrame(CMD_error_values[sorting_ids], CMD_errors_nonan.index[sorting_ids],
                                  CMD_errors_nonan.columns)

        CMD_plx_nonan = self.data[plx_or_d_cols[0]][~nan_ids].to_numpy()
        CMD_plx = CMD_plx_nonan[sorting_ids]

        # Grab the three parameter errors straight from the error columns and convert them to numpy arrays
        delta_m = CMD_errors.filter(regex=self.CMD_specs["filters"][0]).to_numpy().reshape(len(CMD_errors), )
        delta_c1 = CMD_errors.filter(regex=self.CMD_specs["filters"][1]).to_numpy().reshape(len(CMD_errors), )
        delta_c2 = CMD_errors.filter(regex=self.CMD_specs["filters"][2]).to_numpy().reshape(len(CMD_errors), )

        # calculate 1D error for the color axis
        delta_cax = RSS(delta_c1, delta_c2)

        # Exception for delta_cax = 0: Set it to the minimum value that is found in the array
        # zero_indices = np.where(delta_cax is False)[0]
        # print(f"Zero indices for {self.name}: ", zero_indices)
        # m = min(i for i in delta_cax if i > 0)
        # for z_id in zero_indices:
        #    delta_cax[z_id] = m

        # calculate the weights
        try:
            delta_Mabs = abs_mag_error(CMD_plx, CMD_errors[plx_or_d_cols[1]], delta_m).to_numpy()
            weights = (1 / RSS(delta_Mabs, delta_cax)).reshape(len(delta_cax), 1)

        # if no distance errors exist
        except KeyError:
            print("No plx errors found, using only color axis errors for the weight calculation.")
            weights = (1 / delta_cax).reshape(len(delta_cax), 1)

        # all weights need to be greater than zero, then they are scaled and used
        if all(weights >= 0):
            self.weights = min_max_scaler.fit_transform(weights).reshape(len(weights), )

        # otherwise weights are set to unity
        else:
            self.weights = np.ones(len(weights))
            print("No errors found for the CMD data. Setting weight array to unity.")

        if return_deltas:

            # Exception for the case that no distance errors were available
            try:
                return [delta_c1, delta_c2, delta_m, delta_cax, delta_Mabs]
            except UnboundLocalError:
                return [delta_c1, delta_c2, delta_m, delta_cax]

    @staticmethod
    def gridsearch_and_ranking(X_train: np.ndarray, Y_train: np.ndarray, grid: dict, weight_train: np.ndarray,
                               search_function=None, rkf_function=None):
        """
        Perform the hyperparameter gridsearch with 5-fold cross-validation and return the optimized parameters.

        :param X_train: Training data for the first parameter.
        :param Y_train: Training data for the second parameter.
        :param grid: Dictionary containing the grid points that should be evaluated for the three hyperparameters.
        :param weight_train: 1D scalar weights for the SVR.
        :param rkf_function: Fully defined function determining the cross-validation method (default: 5-fold)
        :param search_function: Gridsearch function
                               (default: GridsearchCV, Alternatives e.g., BayesSearchCV, HalvingGridsearchCV)
        :return: Parameters and test scores in the form of a dataframe
        """

        # Create a cross-validation (default: 5-fold)
        if not rkf_function:
            rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)
        else:
            rkf = rkf_function

        # Define the hyperparameter search function (default: GridsearchCV)
        if not search_function:
            search = GridSearchCV(estimator=SVR(), param_grid=grid, cv=rkf)
        else:
            search = search_function(estimator=SVR(), param_grid=grid, cv=rkf)

        # Fit search function to the training data and return a formatted result dataframe
        search.fit(X_train, Y_train, sample_weight=weight_train)
        results_df = pd.DataFrame(search.cv_results_)

        # sort the results by the best performance
        results_df = results_df.sort_values(by=["rank_test_score"])
        results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("kernel")

        return results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

    def SVR_read_from_file(self, file: str):
        """
        Read in the tuned hyperparameters for the specific cluster dataset from the file holding all hyperparameters.

        :param file: Location of the file holding the hyperparameters (csv).
        :return: Dictionary of hyperparameters
        """

        # Open the file containing all saved hyperparameters as dataframe
        hp_df = pd.read_csv(file)

        # Search the entry corresponding to the specified cluster AND dataset AND CMD
        filtered_values = np.where(
            (hp_df['name'] == self.name) & (hp_df["dataset_id"] == self.dataset_id) &
            (hp_df['abs_mag'] == self.CMD_specs['axes'][0]) & (hp_df['cax'] == self.CMD_specs['axes'][1]))[0]

        # Grab only the hyperparameters and convert the line into a dictionary
        params = hp_df.loc[filtered_values][['C', 'epsilon', 'gamma', 'kernel']].to_dict(orient="records")[0]

        return params

    def SVR_Hyperparameter_tuning(self, input_array: np.ndarray, weight_data: np.ndarray, output_file: str = None,
                                  grid: dict = None):
        """
        Tuning of the SVR hyperparameter. Takes the 2D input data along with an 1D weight array, stacks them and calls
        the gridsearch_and_ranking function. After the tuning is finished, the results are written into the global
        hyperparameter file with the specifiers for the cluster, dataset and CMD. As the function is itself called by
        the curve_extraction function, the input_array and weight_data parameters should not be set to object
        attributes.

        :param input_array: 2D input data
        :param weight_data: 1D weight data
        :param output_file: Path to the csv file for collecting all hyperparameters.
        :param grid: (Optional) Dictionary with a custom hyperparameter grid for the gridsearch_and_ranking function
        :return: dictionary of HP parameters
        """

        # Split and reshape the input array for the train_test_split() function
        X_data = input_array[:, 0].reshape(len(input_array[:, 0]), 1)

        # Sample weights are passed with the Y-values to uphold the right value-weight combinations in the split
        Y_data = np.stack([input_array[:, 1], weight_data], axis=1)

        # Define a training and test set
        X_tr, X_test, Y_tr, Y_test = train_test_split(X_data, Y_data, random_state=13)

        # Scale training set
        X_mean = np.mean(X_tr)
        X_std = np.std(X_tr)
        X_train_scaled = (X_tr - X_mean) / X_std

        # De-tangle the Y-data and weights of the newly defined training set again
        Y_flat = Y_tr[:, 0].ravel()
        Y_weights = Y_tr[:, 1].copy(order="C")  # make the weight array C-writeable (Python bug)

        # Create parameter grid (These are the tested grid values from the Working_SVR_isochrones.py file)
        if grid is None:
            evals = np.logspace(-2, -1.5, 20)
            Cvals = np.logspace(-2, 2, 20)
            grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals, epsilon=evals)

        # Call the gridsearch function
        ranking = self.gridsearch_and_ranking(X_train=X_train_scaled, Y_train=Y_flat, grid=grid, weight_train=Y_weights)
        print("...finished tuning")

        # Cluster name and CMD specs MUST be included for unique identification of the correct hyperparameters
        output_data = {"name": self.name, "abs_mag": self.CMD_specs["axes"][0], "cax": self.CMD_specs["axes"][1],
                       "score": ranking.mean_test_score[0], "std": ranking.std_test_score[0],
                       "dataset_id": self.dataset_id}

        # Write the output to the hyperparameter file:
        if output_file:
            output_data = output_data | ranking.params[0]
            df_row = pd.DataFrame(data=output_data, index=[0])
            df_row.to_csv(output_file, mode="a", header=False)

        else:
            print(ranking.params[0])
            print("score:", ranking.mean_test_score[0])
            print("std:", ranking.std_test_score[0])

        # Return params for check
        return ranking.params[0]

    def curve_extraction(self, svr_data: np.ndarray, HP_file: str, svr_predict: np.ndarray,
                         svr_weights: np.ndarray, grid: dict = None, always_tune: bool = False):
        """
        Read in hyperparameters if possible, else call the SVR_hyperparameter_tuning function. Then calculate the
        regression curve using the hyperparameters for the svr_data (original or bootstrapped array) and transform
        the result back into the CMD space.

        :param svr_data: Array to be subjected to regression. Either original PCA data or bootstrapped data
        :param HP_file: Path to the csv file containing the hyperparameters (same as in SVR_read_from_file)
        :param svr_predict: Array used for the prediction, i.e. the array upon which the regression will be done after training the model. If the svr_data is an array of bootstrapped values, the prediction still needs to be performed on the original X data, otherwise the confidence borders will not be smooth
        :param svr_weights: Array containing the weight data, either original or bootstrapped
        :param grid: Dictionary of custom grid if wanted
        :param always_tune: Flag that forces the algorithm to tune the hyperparameters, even if data already exists in the HP file
        :return: PCA curve and isochrone
        """

        # Define array for the prediction of the curve
        #

        # Reshape the prediction array (sklearn bug)
        svr_predict = svr_predict.reshape(len(svr_predict), 1)

        # Either read in the SVR parameters from the HP file, or determine them by tuning first
        if not always_tune:
            try:
                params = self.SVR_read_from_file(file=HP_file)
            except IndexError:
                print("Index error: Running HP tuning for {}...".format(self.name))
                params = self.SVR_Hyperparameter_tuning(input_array=svr_data, weight_data=svr_weights,
                                                        output_file=HP_file, grid=grid)
        else:
            print(f"The always_tune flag is set. Starting tuning for {self.name} ...")
            params = self.SVR_Hyperparameter_tuning(input_array=svr_data, weight_data=svr_weights, output_file=None,
                                                    grid=grid)

        # Define the two coordinates for SVR and fit-predict the tuned model to them
        X = svr_data[:, 0].reshape(len(svr_data[:, 0]), 1)
        Y = svr_data[:, 1]
        svr_model = SVR(**params)
        Y_all = svr_model.fit(X, Y, sample_weight=svr_weights).predict(svr_predict)

        # The results are a PCA curve and the corresponding isochrone
        curve = np.stack([svr_predict[:, 0], Y_all], 1)
        rev_transform = self.pca.inverse_transform(curve)

        return curve, rev_transform

    def resample_curves(self, idx: int, output: np.ndarray, sampling_array: np.ndarray = None,
                        sampling_weights: np.ndarray = None, kwargs: dict = None):
        """
        Function that runs in parallel and computes bootstrapped datasets which it passes to the curve_extraction
        routine to produce resampled isochrones. Then it stores the results at the given index in the output array.

        :param idx: Bootstrapping index passed by the parallel function that allocates the location in the output
                    array where the results will be saved for the current resampling
        :param output: Dataframe for collecting all resampling results that are computed in parallel
        :param sampling_array: Array from which to bootstrap (Default: self.PCA_XY)
        :param sampling_weights: Weights belonging to the sampling array (Default: self.weights)
        :param kwargs: passes grid and "always_tune" flag command through to curve_extraction
        :return: Write output in the output file at the assigned index
        """

        # If no sampling array or weights are given, use the cluster attributes PCA_XY and weights
        if sampling_array is None:
            sampling_array = self.PCA_XY
        if sampling_weights is None:
            sampling_weights = self.weights

        # Stack the XY array and the weights to allow for joint resampling
        XY_weights_stack = np.stack([sampling_array[:, 0], sampling_array[:, 1], sampling_weights], axis=1)

        # resample the dataset for the SVR regression
        bs = resample(XY_weights_stack)

        # Split up the resampled array again
        bs_XY, bs_weights = bs[:, :2].copy(order="C"), bs[:, 2].copy(order="C")

        # Calculate the PCA curve / isochrone from the bootstrapped data, using the original X_array as prediction data
        curve, isochrone = self.curve_extraction(svr_data=bs_XY, svr_weights=bs_weights,
                                                 svr_predict=sampling_array[:, 0], **kwargs)

        # Write the result in the provided output file
        output[:, :2, idx] = curve
        output[:, 2:4, idx] = isochrone

    def interval_stats(self, n_resample: int, original_array=None, original_weights=None, njobs: int = None,
                       kwargs: dict = None):
        """
        The original dataset is provided to the resampling function, which is called in parallel processing and its
        results are stored in the output array defined here. From this array the median and 5th/95th percentile curves
        are calculated.

        :param n_resample: Number of resampled isochrones to generate (typically 100 - 1000)
        :param original_array: Array that should be used for bootstrapping and resampling
        :param original_weights: Weights corresponding to this array
        :param njobs: Number of parallel jobs
        :param kwargs: Passing kwargs for the other functions in the function tree (= grid dict and always_tune flag)
        :return: Empirical isochrone and 5/95 perc bounds
        """

        # Set generic number of parallel jobs
        if not njobs:
            njobs = 6

        # Create output array where all resampled curves will be stowed
        isochrone_array = np.empty(shape=(len(self.PCA_XY[:, 0]), 4, n_resample))

        # Parallelized generation of the resampled curves
        tic = time.time()
        Parallel(n_jobs=njobs, require="sharedmem")(
            delayed(self.resample_curves)(idx, output=isochrone_array, sampling_array=original_array,
                                          sampling_weights=original_weights, kwargs=kwargs) for idx in
            range(n_resample))
        toc = time.time()
        elapsed_time = toc - tic
        print(f"The resampling of {n_resample} curves took {elapsed_time} s ({njobs} parallel jobs).")

        # Create an array holding the stats and walk through all possible X-values(= svr_predict[:,0] array)
        stats_array = np.empty(shape=(len(isochrone_array[:, 0, :]), 3))

        # Walk through the X-axis values
        for j, x_i in enumerate(self.PCA_XY[:, 0]):
            PCA_y_vals = []

            # Over all the resampled curves, collect the Y values corresponding to the current X value
            # (there could be multiple Ys for each X due to the bootstrapping)
            for i in range(n_resample):
                PCA_y_vals.extend(isochrone_array[ii, 1, i] for ii in np.where(isochrone_array[:, 0, i] == x_i)[0])

            # Calculate the median and percentiles for each X value from all bootstrapped curves
            PCA_y_median = np.median(PCA_y_vals)
            PCA_y_lower, PCA_y_upper = np.percentile(PCA_y_vals, [5, 95])

            # Write the result into the corresponding row of the stats array
            stats_array[j, :] = PCA_y_lower, PCA_y_median, PCA_y_upper

        return stats_array

    def isochrone_and_intervals(self, n_boot: int, data: np.ndarray = None, weights: np.ndarray = None,
                                parallel_jobs: int = None, output_loc: str = None, kwargs: dict = None):
        """

        :param n_boot: Number of resamplings
        :param data: Array that should be resampled
        :param weights: Corresponding weights
        :param parallel_jobs: Number of parallel jobs
        :param output_loc: Path to the directory where the result should be saved
        :param kwargs: Keyword arguments to be passed to other functions in the function tree
                       (= grid dict and always_tune flag)
        :return: Dataframe containing the empirical isochrone and the uncertainty bounds
        """

        # Define x_array for the stacking and reverse transformation if no data array is given
        if data is None:
            x_array = self.PCA_XY[:, 0]
        else:
            x_array = data[:, 0]

        # Call the interval_stats() function to get the PCA stats data for the Y variable
        y_array = self.interval_stats(n_resample=n_boot, original_array=data, original_weights=weights,
                                      njobs=parallel_jobs, kwargs=kwargs)

        # Initialize result array (three XY-pairs)
        sorted_array = np.empty(shape=(len(x_array), 6))
        col_counter = 0

        # Stack the X array to each of the Y arrays and transform the coordinate pairs back into CMD curves
        for col in range(3):
            stack = np.stack([x_array, y_array[:, col]], axis=1)
            rev_stack = self.pca.inverse_transform(stack)
            sorted_stack = rev_stack[rev_stack[:, 1].argsort()]
            sorted_array[:, col_counter:col_counter + 2] = sorted_stack
            col_counter += 2

        # Create a dataframe from the returned results
        new_cols = ["l_x", "l_y", "m_x", "m_y", "u_x", "u_y"]
        sorted_df = pd.DataFrame(data=sorted_array, columns=new_cols)

        # Save the dataframe as csv file
        if output_loc:
            output_file = "{0}_{1}_nboot_{2}_cat_{3}.csv".format(self.name, self.CMD_specs["short"], n_boot,
                                                                 self.dataset_id)
            sorted_df.to_csv(output_loc + output_file, mode="w", header=True)

        return sorted_df
