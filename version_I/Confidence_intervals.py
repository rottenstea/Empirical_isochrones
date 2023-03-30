from version_I.Support_Vector_Regression import *
# for bootstrapping
from sklearn.utils import resample


# Confidence intervals
# -------------------------------------------

def array_sorting(A: list, index: int = 1):
    sorted_elements = []
    for element in A:
        sorted_elements.append(element[element[:, index].argsort()])
    return sorted_elements


def Isochrone_collection(reference_isochrone, bs_array, n_boot, method_kwargs: dict):#, pca_case: bool = False):
    #if pca_case:
    isochrone_store = np.empty(shape=(len(reference_isochrone[:, 0]), 4, n_boot))
    #else:
    #isochrone_store = np.empty(shape=(len(reference_isochrone[:, 0]), 2, n_boot))

    for i in range(n_boot):
        print("i=", i + 1)
        bs = resample(bs_array, n_samples=(len(reference_isochrone[:, 0])))
        #if pca_case:
        pca_isos, real_isos = SVR_PCA_calculation(input_arr=bs, **method_kwargs)
        isochrone_store[:, 0, i] = pca_isos[:, 0]
        isochrone_store[:, 1, i] = pca_isos[:, 1]
        isochrone_store[:, 2, i] = real_isos[:, 0]
        isochrone_store[:, 3, i] = real_isos[:, 1]
        # else:
        #     real_isos = SVR_PCA_calculation(input_arr=bs, **method_kwargs)
        #     isochrone_store[:, 0, i] = real_isos[:, 0]
        #     isochrone_store[:, 1, i] = real_isos[:, 1]

    return isochrone_store


def Confidence_interval_stats(reference_isochrone, isochrone_collection, pca_case: bool = False, pca_func=None):
    stats_array = np.empty(shape=(len(reference_isochrone[:, 0]), 3))
    n_iter = len(isochrone_collection[0, 0, :])
    for j, x_i in enumerate(reference_isochrone[:, 0]):

        idxes = []
        y_vals = []
        for i in range(n_iter):
            #print(i)
            try:
                ii = np.where(isochrone_collection[:, 0, i] == x_i)[0]
                for el in ii:
                    idxes.append(el)
                    y_vals.append(isochrone_collection[el, 1, i])  # x_vals are identical anyway (i checked)
            except ValueError:
                # print(x_i," not in list in", i)
                pass
        # print(idxes)

        y_median = np.median(y_vals)
        y_lower, y_upper = np.percentile(y_vals, [5, 95])
        stats_array[j, :] = y_lower, y_median, y_upper

    if pca_case:
        l = np.stack([reference_isochrone[:, 0], stats_array[:, 0]], axis=1)
        m = np.stack([reference_isochrone[:, 0], stats_array[:, 1]], axis=1)
        u = np.stack([reference_isochrone[:, 0], stats_array[:, 2]], axis=1)

        lower_r = pca_func.inverse_transform(l)
        median_r = pca_func.inverse_transform(m)
        upper_r = pca_func.inverse_transform(u)

        lower, upper, median = array_sorting([lower_r, upper_r, median_r], 0)
        return lower, upper, median

    else:
        return stats_array
        # np.stack([reference_isochrone[:, 0], stats_array[:, 0]], axis=1), \
        # np.stack([reference_isochrone[:, 0], stats_array[:, 2]], axis=1), \
        # np.stack([reference_isochrone[:, 0], stats_array[:, 1]], axis=1)
