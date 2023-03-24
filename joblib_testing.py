from Classfile import *
import os
from joblib import dump, load, Parallel, delayed
import time

""" This file contains a short script that shows the function of using the joblib python package for a parallelization 
of a short resampling function of my isochrones. The script has been integrated into the isochrone generation functions
in an altered fashion (without dump and load), but I keep this script to try out further stuff and as a reminder."""

if __name__ == "__main__":

    # load pre-processing function for my different catalogs
    from pre_processing import create_df

    # if using load and dump it is best to have this folder for the memmaps
    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    # load cluster data as formatted data frame
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

    #  kwargs for the HP file and HP search grid
    kwargs = {"HP_file": "data/Hyperparameters/CatalogII.csv", "grid": None}

    # define number of resamplings
    n_boot = 20

    for cluster in CII_clusters[2:3]:
        OC = star_cluster(cluster, CII_df)
        OC.create_CMD()

        # -------------------------------------------------------------------------------------------------------------
        # PARALLEL

        data = OC.PCA_XY

        # dump data in memmap
        data_filename_memmap = os.path.join(folder, '{}_data_memmap'.format(OC.name))
        dump(data, data_filename_memmap)

        # load data from memmap
        data = load(data_filename_memmap, mmap_mode='r')

        # create output memmap
        output_filename_memmap = os.path.join(folder, '{}_output_memmap'.format(OC.name))
        output = np.memmap(output_filename_memmap, dtype=data.dtype,
                           shape=(len(data[:, 0]), 4, n_boot), mode='w+')

        # time the function runtime
        tic = time.time()
        Parallel(n_jobs=6)(delayed(OC.resample_curves)(data, output, idx, **kwargs) for idx in range(n_boot))
        toc = time.time()
        print(toc - tic, "s parallel")

        # convert output to np.array for further use
        isochrone_store = np.array(output)
        # -------------------------------------------------------------------------------------------------------------
        # SERIAL
        #
        # define output array
        # isochrone_store = np.empty(shape=(len(OC.CMD[:, 0]), 4, n_boot))

        # time function runtime in serial fashion
        # tic1 = time.time()
        # for i in range(n_boot):
        #    resample_recal(OC, OC.PCA_XY, weights, isochrone_store, i)

        # toc1 = time.time()
        # print(toc1 - tic1, "s serial")

        # -------------------------------------------------------------------------------------------------------------
        # Benchmarks: with 8 jobs and n_boot = 1000
        # 11.705401182174683 s parallel
        # 31.09577512741089  s serial

        # plot the results

        f = CMD_density_design(OC.CMD, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_store[:, 2, i], isochrone_store[:, 3, i], color="orange")
        f.show()

        g = CMD_density_design(OC.PCA_XY, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_store[:, 0, i], isochrone_store[:, 1, i], color="orange")
        g.show()
