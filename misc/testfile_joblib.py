from Extraction.Classfile import *
from My_tools import my_utility
import os
from joblib import dump, load, Parallel, delayed
import time
import matplotlib.pyplot as plt

""" This file contains a short script that shows the function of using the joblib python package for a parallelization 
of a short resampling function of my isochrones. The script has been integrated into the isochrone generation functions
in an altered fashion (without dump and load), but I keep this script to try out further stuff and as a reminder."""

if __name__ == "__main__":

    # load pre-processing function for my different catalogs
    from Extraction.pre_processing import cluster_df_list, cluster_name_list

    HP_file = "/Users/alena/PycharmProjects/PaperI/data/Hyperparameters/Archive_full.csv"
    my_utility.setup_HP(HP_file)

    # if using load and dump it is best to have this folder for the memmaps
    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    # load cluster data as formatted data frame
    CII_clusters, CII_df = cluster_name_list[1], cluster_df_list[1]

    #  kwargs for the HP file and HP search grid
    kwargs = dict(grid=None, HP_file=HP_file, catalog_mode=True)

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
        Parallel(n_jobs=6)(delayed(OC.resample_curves)(idx=idx, output=output, kwargs=kwargs) for idx in range(n_boot))
        toc = time.time()
        print(toc - tic, "s parallel")

        # convert output to np.array for further use
        isochrone_store = np.array(output)

        f = CMD_density_design(OC.CMD, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_store[:, 2, i], isochrone_store[:, 3, i], color="orange")
        f.show()

        g = CMD_density_design(OC.PCA_XY, cluster_obj=OC)
        for i in range(n_boot):
            plt.plot(isochrone_store[:, 0, i], isochrone_store[:, 1, i], color="orange")
        g.show()
