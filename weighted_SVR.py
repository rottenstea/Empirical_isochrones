import pandas as pd
import numpy as np

# Support vector regression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from itertools import islice
import ast

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold


# Support vector regression
# -------------------------------------------

# gridsearch function that creates the ranking
# CHECK 16.02.2022 ---> works
def gridsearch_and_ranking(X_train, Y_train, pg, rkf, weights):
    search = GridSearchCV(estimator=SVR(), param_grid=pg, cv=rkf)

    search.fit(X_train, Y_train, sample_weight=weights)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    return results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]


# Currently in classfile but kept for later possibly
# ----------------------------------------------------------------------------------------------------------------------
def SVR_read_from_file(cluster_ob, param_file):
    hp_df = pd.read_csv(param_file)

    filtered_values = np.where(
        (hp_df['name'] == cluster_ob.name) & (hp_df['abs_mag'] == cluster_ob.CMD_specs['axes'][0]) &
        (hp_df['cax'] == cluster_ob.CMD_specs['axes'][1]))[0]

    params = hp_df.loc[filtered_values][['C', 'epsilon', 'gamma', 'kernel']].to_dict(orient="records")[0]
    return params


def SVR_Hyperparameter_tuning(cluster_ob, array, grid_dict: dict = None, weight_data: np.array = None,
                              output_file=None):
    if weight_data is None:
        weight_data = np.ones(shape=(len(array[:, 1])))

    X_data = array[:, 0].reshape(len(array[:, 0]), 1)
    Y_data = np.stack([array[:, 1], weight_data], axis=1)

    # 2. Split X and Y into training and test set
    X_tr, X_test, Y_tr, Y_test = train_test_split(X_data, Y_data, random_state=13)

    # 3. Scale training and test set wrt. to the training set
    X_mean = np.mean(X_tr)
    X_std = np.std(X_tr)
    X_train_scaled = (X_tr - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    X_scaled = np.array((X_data - X_mean) / X_std)
    Y_flat = Y_tr[:, 0].ravel()
    Y_err = Y_tr[:, 1].copy(order="C")

    # 4. Define 5-fold cross validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)

    # 5. Create parameter grid
    if grid_dict is None:
        kernels = ["rbf"]
        C_range = np.logspace(-1, 2, 10)
        epsilon_range = np.logspace(-6, 1, 10)

        grid = [
            dict(kernel=kernels, gamma=["auto"], C=C_range,
                 epsilon=epsilon_range), ]

    else:
        grid = [grid_dict, ]

    # 6. call the gridsearch function
    ranking = gridsearch_and_ranking(X_train=X_train_scaled, Y_train=Y_flat, pg=grid, rkf=rkf, weights=Y_err)
    print("fin")

    # 7. Write output to file
    if output_file:
        output_data = {"name": cluster_ob.name, "abs_mag": cluster_ob.CMD_specs["axes"][0],
                       "cax": cluster_ob.CMD_specs["axes"][1],
                       "score": ranking.mean_test_score[0], "std": ranking.std_test_score[0]}
        output_data = output_data | ranking.params[0]

        df_row = pd.DataFrame(data=output_data, index=[0])
        df_row.to_csv(output_file, mode="a", header=False)

    else:
        print(ranking.params[0])
        print("score:", ranking.mean_test_score[0])
        print("std:", ranking.std_test_score[0])

    # 8. return params for check
    return ranking.params[0]



#def Empirical_isochrone(n_boot: int):

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    from Classfile import *

    from pre_processing import create_df
    from sklearn.decomposition import PCA

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
        OC.create_CMD(CMD_params=["Gmag", "G-RP"])

        params_w = OC.SVR_read_from_file(HP_file="data/Hyperparameters/CatalogII.csv")

        weights = OC.create_weights()

        pca = PCA(n_components=2)
        pca_arr = pca.fit_transform(OC.CMD)

        svr_predict = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)

        X = pca_arr[:, 0].reshape(len(pca_arr[:, 0]), 1)
        Y = pca_arr[:, 1]

        evals = np.logspace(-4, -2, 10)
        Cvals = np.logspace(-2, 2, 10)

        param_grid = dict(kernel=["rbf"], gamma=["scale"], C=Cvals,
                          epsilon=evals)

        # params = {'C': 14.38449888287663, 'epsilon ': 0.02801356761198867, 'gamma': 'scale', 'kernel': 'rbf'}

        # -----------------------------------------------------------------------------------------------------------------
        '''
        # Case 1 : No weights

        params_u = SVR_Hyperparameter_tuning(pca_arr, param_grid,output_file="data/Hyperparameters/CatalogII.csv", cluster_obj=OC)

        print("unweighted params", params_u)
        svr_u = SVR(**params_u)

        Y_u = svr_u.fit(X, Y).predict(svr_predict)

        print("SVR Test score:", svr_u.score(svr_predict, Y.ravel()))

        SVR_u = np.stack([svr_predict[:, 0], Y_u], 1)
        SVR_u = SVR_u[SVR_u[:, 0].argsort()]
        rev_u = pca.inverse_transform(SVR_u)
        '''
        # -----------------------------------------------------------------------------------------------------------------

        # Case 2 : with weights

        # params_w = SVR_Hyperparameter_tuning(pca_arr, param_grid, weight_data=weights,
        #                                      output_file="data/Hyperparameters/CatalogII.csv", cluster_obj=OC)
        # print("weighted params", params_w)

        svr_w = SVR(**params_w)

        Y_w = svr_w.fit(X, Y, sample_weight=weights).predict(svr_predict)
        SVR_w = np.stack([svr_predict[:, 0], Y_w], 1)
        SVR_w = SVR_w[SVR_w[:, 0].argsort()]
        rev_w = pca.inverse_transform(SVR_w)

        # -----------------------------------------------------------------------------------------------------------------

        # Plot the results

        kr = 0
        fig1 = plt.figure(figsize=(4, 6))
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.scatter(OC.density_x, OC.density_y, label=OC.name, **OC.kwargs_CMD)

        # ax.plot(rev_u[kr:, 0], rev_u[kr:, 1], color="red", label="unweighted")
        ax.plot(rev_w[kr:, 0], rev_w[kr:, 1], color="orange", ls="-", label="weighted")
        # ax2.set_ylabel(r"M$_{\rm G}$")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)

        ax.set_ylabel(OC.CMD_specs["axes"][0])
        ax.set_xlabel(OC.CMD_specs["axes"][1])

        ax.set_title("Empirical isochrone")  # , y=0.97)
        # a = 100
        # plt.plot(i[:a] - K[:a], i[:a], color="orange", label = "PARSEC")

        plt.legend(bbox_to_anchor=(1, 1), loc="upper right")

        fig1.show()
