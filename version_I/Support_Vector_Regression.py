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
def gridsearch_and_ranking(X_train, Y_train, pg, rkf, weight_data = None):
    search = GridSearchCV(estimator=SVR(), param_grid=pg, cv=rkf)

    search.fit(X_train, Y_train, sample_weight=weight_data)

    #search.fit(X_train, Y_train)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    return results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]


# reading in the best hyperparameters for each cluster and returning them
# as list
# CHECK 16.02.2022 ---> works
def SVR_read_from_file(param_file="data/SVR_params/best_params_comp.txt"):
    with open(param_file) as f:
        param_list = list(islice(f, None))
        param_list_clean = [element.strip() for element in param_list]

    return param_list_clean


def SVR_Hyperparameter_tuning(array, grid_dict: dict = None, output_file=None, further_data: dict = None, weight_data = None):

    #if weight_data is None:
     #   weight_data = np.ones(len(array[:,0]))

    X = array[:, 0].reshape(len(array[:, 0]), 1)

    Y = np.stack([array[:, 1], weight_data], axis=1)

    # 2. Split X and Y into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=13)

    Y_flat = Y_train[:, 0].ravel()
    Y_err = Y_train[:, 1].copy(order="C")

    # 2. Split X and Y into training and test set
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=13)

    # 3. Scale training and test set wrt. to the training set
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    X_scaled = np.array((X - X_mean) / X_std)
    #Y_flat = Y_train.ravel()

    # 4. Define 5-fold cross validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=13)

    # 5. Create parameter grid
    if grid_dict is None:
        kernels = ["rbf"]
        C_range = np.logspace(-1, 2, 10)
        # gamma_range = np.logspace(-5, 2, 8)
        epsilon_range = np.logspace(-6, 1, 10)

        # svr_grid_rbf = dict(kernel=kernels, gamma=gamma_range, C=C_range)

        grid = [
            dict(kernel=kernels, gamma=["auto"], C=C_range,
                 epsilon=epsilon_range), ]

    else:
        grid = [grid_dict, ]

    #Y_flat = Y_train.ravel()

    # 6. call the gridsearch function
    ranking = gridsearch_and_ranking(X_train_scaled, Y_flat, grid, rkf, weight_data = Y_err)
    # print("fin")

    # 7. Write output to file
    if output_file:
        df_row = pd.DataFrame(ranking.params[0], index=[0])

        df_row.insert(0, "score", ranking.mean_test_score[0], True)
        df_row.insert(0, "std", ranking.std_test_score[0], True)

        if further_data:
            for key in further_data.keys():
                val = further_data[key]
                df_row.insert(0, key, val, True)

        df_row.to_csv(output_file, mode="a", header=False)

        # with open(output_file, 'w') as f:
        #   print(ranking.params[0],file = f)
    else:
        print(ranking.params[0])
        print("score:", ranking.mean_test_score[0])
        print("std:", ranking.std_test_score[0])

    # 8. return params for check
    return ranking.params[0]


# standard SVR calculation
def SVR_calculation(SVR_data, SVR_GS_params: list, sample_name: str, SVR_predict=None):
    if SVR_predict is None:
        SVR_predict = SVR_data[:, 0].reshape(len(SVR_data[:, 0]), 1)
    else:
        SVR_predict = SVR_predict[:, 0].reshape(len(SVR_predict[:, 0]), 1)

    X = SVR_data[:, 0].reshape(len(SVR_data[:, 0]), 1)
    Y = SVR_data[:, 1]
    start = SVR_GS_params.index(sample_name)
    # print(start, sample_name)
    param_str, SVR_score, SVR_std = SVR_GS_params[start + 1], SVR_GS_params[start + 2], SVR_GS_params[start + 3]
    param_dict = ast.literal_eval(param_str)
    # print(param_dict)
    SVR_model_all = SVR(**param_dict)
    SVR_model_all.fit(X, Y.ravel())
    Y_pred_all = SVR_model_all.predict(SVR_predict)
    # print("SVR Test score:", SVR_model_all.score(SVR_predict, Y.ravel()))
    # switched for testing
    SVR_all = np.sort(np.stack([SVR_predict.ravel(), Y_pred_all], 1), axis=0)
    return SVR_all, SVR_score, SVR_std


# SVR_PCA_calculation 2-in-1

def SVR_PCA_calculation(input_arr, keys, sample_name, pca_case: bool = False, pca_func=None, svr_predict=None):
    if svr_predict is None:
        svr_predict = input_arr[:, 0].reshape(len(input_arr[:, 0]), 1)
    else:
        svr_predict = svr_predict[:, 0].reshape(len(svr_predict[:, 0]), 1)

    start = keys.index(sample_name)
    param_str, SVR_score, SVR_std = keys[start + 1], keys[start + 2], keys[start + 3]
    param_dict = ast.literal_eval(param_str)
    X = input_arr[:, 0].reshape(len(input_arr[:, 0]), 1)
    Y = input_arr[:, 1]

    SVR_model_all = SVR(**param_dict)
    Y_all = SVR_model_all.fit(X, Y).predict(svr_predict)
    # print("SVR Test score:", SVR_model_all.score(svr_predict, Y.ravel()))

    SVR_all = np.stack([svr_predict[:, 0], Y_all], 1)

    if pca_case:
        rev_transform = pca_func.inverse_transform(SVR_all)
        return SVR_all, rev_transform  # [rev_transform[:, 1].argsort()]
    else:
        return np.sort(SVR_all, axis=0)  # , SVR_score, SVR_std
