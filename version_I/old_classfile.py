import numpy as np
import pandas as pd

from version_I.Support_Vector_Regression import gridsearch_and_ranking

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

from Plotting_essentials import CMD_density_design


class star_cluster:

    # UPDATE & RE-DESIGN 09.12.22 ---> works
    def __init__(self, name: str, catalog: pd.DataFrame, CMD_parameters: list = None,
                sort_idx: int = 1, errors: list = None, index_only: bool = False,
                 phot_sys=None):

        # photometric system option
        if phot_sys is None:
            phot_sys = ["gaia"]
        if CMD_parameters is None:
            CMD_parameters = ["Gmag", "BP-RP"]

        # raw data
        # slashes in clusternames are causing problems
        if "/" in name:
            self.name = name.replace("/", "_")
        else:
            self.name = name

        self.data = catalog[catalog.Cluster_id == name]

        # astrometric columns
        self.parallax = self.data.plx

        # photometric columns
        for system in phot_sys:
            if system == "gaia":
                self.G = self.data.Gmag
                self.BP_RP = self.data["BP-RP"]
                if not index_only:
                    self.BP = self.data.BPmag
                    self.RP = self.data.RPmag
#                    self.c1 = self.blue - self.G
#                    self.c2 = self.G - self.red
 #               else:
 #                   self.c1 = None
 #                   self.c2 = None

                # color index options

#                color_dict = {1: self.c1, 2: self.c2, 3: self.c3}
#                self.cax = color_dict[color_index]

                # error options
                if errors:
                    self.G_error = self.data.e_Gmag
                    self.BP_error = self.data.e_BPmag
                    self.RP_error = self.data.e_RPmag
                    self.plx_error = self.data.e_plx

            elif system == "panstarrs":
                self.u = self.data.u_mag
                self.g = self.data.g_mag
                self.i = self.data.i_mag
                self.r = self.data.r_mag
                self.Y = self.data.Y_mag

            elif system == "2mass":
                self.J = self.data.J_mag
                self.H = self.data.H_mag
                self.K = self.data.K_mag


        # color index and abs mag
        if len(CMD_parameters) == 2:
            self.abs_mag_filter, self.cax = self.data[CMD_parameters[0]], self.data[CMD_parameters[1]]
        elif len(CMD_parameters) == 3:
            abs_mag_filter, cax1, cax2 = self.data[CMD_parameters[0]], self.data[CMD_parameters[1]], self.data[CMD_parameters[2]]
            self.cax = cax1 - cax2


        # CMD data calculation
        self.distance = 1000 / self.parallax
        self.abs_mag = (self.G - 5 * np.log10(self.distance) + 5)

        arr = np.stack([self.cax, self.abs_mag], axis=1)
        cleaned_arr = arr[~np.isnan(arr).any(axis=1), :]
        sorted_arr = cleaned_arr[cleaned_arr[:, sort_idx].argsort()]

        self.CMD = sorted_arr
        self.Nstars = len(self.CMD)

        # PCA variables
        self.PCA_medians = None
        self.PCA_XY = None
        self.PCA_isochrone = None

        # SVR Variables
        self.SVR_isochrone = None
        self.isochrone_array = None
        self.CI_stats = None
        self.conv_percentile_isochrone = None
        self.percentile_isochrone = None

        # Plotting variables
        self.density_profile = None
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)







# moved to Support_Vector_Regression.py (18-1-23)

    # NOT CHECKED YET
    def SVR_Hyperparameter_tuning(self, file_path: str, PCA: bool = False, pca_array=None, grid_dict: dict = None):

        f = open(file_path, "w")

        # 1. Define explanatory and response variables
        if PCA:
            X = pca_array[:, 0].reshape(len(pca_array[:, 0]), 1)
            Y = pca_array[:, 1]
        else:
            X = self.CMD[:, 0].reshape(len(self.CMD[:, 0]), 1)
            Y = self.CMD[:, 1]

        # 2. Split X and Y into training and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=13)

        # 3. Scale training and test set wrt. to the training set
        X_mean = np.mean(X_train)
        X_std = np.std(X_train)
        X_train_scaled = (X_train - X_mean) / X_std
        X_test_scaled = (X_test - X_mean) / X_std
        X_scaled = np.array((X - X_mean) / X_std)
        Y_flat = Y_train.ravel()

        # 4. Define 5-fold cross validation
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)

        # 5. Create parameter grid
        if grid_dict is None:
            kernels = ["rbf"]
            C_range = np.logspace(-3, 2, 8)
            gamma_range = ["auto"]#np.logspace(-5, 2, 8)
            epsilon_range = np.logspace(-5, -2, 8)

            #svr_grid_rbf = dict(kernel=kernels, gamma=gamma_range, C=C_range)

            grid = [
                dict(kernel=kernels, gamma=gamma_range, C=C_range,
                     epsilon=epsilon_range), ]

        else:
            grid = [grid_dict, ]

        Y_flat = Y_train.ravel()

        # 6. call the gridsearch function
        ranking = gridsearch_and_ranking(X_train_scaled, Y_flat, grid, rkf)
        print("fin")

        # 7. Write output to file
        print(self.name, file=f)
        print(ranking.params[0], file=f)
        print("score:", ranking.mean_test_score[0], file=f)
        print("std:", ranking.std_test_score[0], file=f)

        # 8. return params for check
        return ranking.params[0]

'''

class star_cluster:

    # UPDATE & RE-DESIGN 09.12.22 ---> works
    def __init__(self, name: str, catalog: pd.DataFrame, CMD_parameters: list = None,
                 sort_idx: int = 1, errors: list = None):

        # raw data
        # slashes in clusternames are causing problems
        if "/" in name:
            self.name = name.replace("/", "_")
        else:
            self.name = name

        # photometric system option
        if CMD_parameters is None:
            CMD_parameters = ["G_mag", "BP_RP"]

        self.data = catalog[catalog.Cluster_id == name]

        # color index and abs mag
        if len(CMD_parameters) == 2:
            abs_mag_filter, self.cax = self.data[CMD_parameters[0]], self.data[CMD_parameters[1]]
        elif len(CMD_parameters) == 3:
            abs_mag_filter, cax1, cax2 = self.data[CMD_parameters[0]], \
                                         self.data[CMD_parameters[1]], self.data[CMD_parameters[2]]
            self.cax = cax1 - cax2
        else:
            abs_mag_filter = None
            self.cax = None

        # if there are errors, collect the subset of df columns
        if errors:
            self.error_array = self.data[errors]

        self.age = None
        self.Catalog = None

        # CMD data calculation
        self.distance = 1000 / self.data.plx
        self.abs_mag = (abs_mag_filter - 5 * np.log10(self.distance) + 5)

        arr = np.stack([self.cax, self.abs_mag], axis=1)
        cleaned_arr = arr[~np.isnan(arr).any(axis=1), :]
        sorted_arr = cleaned_arr[cleaned_arr[:, sort_idx].argsort()]

        self.CMD = sorted_arr
        self.Nstars = len(self.CMD)

        # PCA variables
        self.PCA_medians = None
        self.PCA_XY = None
        self.PCA_isochrone = None

        # SVR Variables
        self.SVR_isochrone_x = None
        self.SVR_isochrone_y = None
        self.isochrone_array = None
        self.CI_stats = None
        self.conv_percentile_isochrone = None
        self.percentile_isochrone = None

        # Plotting variables
        self.density_profile = None
        self.density_x, self.density_y, self.kwargs_CMD = CMD_density_design([self.CMD[:, 0], self.CMD[:, 1]],
                                                                             density_plot=False)
'''