import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.stats import gaussian_kde
import seaborn as sns


# Density map and plots of the clusters
# -------------------------------------------
def CMD_density_design(A, gamma: float = 0.7, to_RBG: list = None, from_RBG: list = None, marker: str = ".",
                       s: int = 50, lw: int = 0, density_plot=True, kde: bool = False, cluster_obj=None,
                       fig_specs: list = None, title_axes_specs: list = None):
    if from_RBG is None:
        from_RBG = [0.62, 0.79, 0.88]
    if to_RBG is None:
        to_RBG = [0.0, 0.25, 0.53]  # sign blue https://www.december.com/html/spec/colorper.html

    if type(A) == list:
        A_stack = np.vstack([A[0], A[1]])
    else:
        A_stack = np.transpose(A)

    kde_A = gaussian_kde(A_stack)(A_stack)
    arg_idx = kde_A.argsort()
    x, y, z = A_stack[0, arg_idx], A_stack[1, arg_idx], kde_A[arg_idx]

    r1, g1, b1 = from_RBG
    r2, g2, b2 = to_RBG

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cluster_map = LinearSegmentedColormap('custom_cmap', cdict)

    kwargs_CMD = dict(c=z, cmap=cluster_map, marker=marker, s=s, lw=lw, norm=PowerNorm(gamma))

    if density_plot:
        sns.set_style("darkgrid")
        if not fig_specs:
            fig_specs = [4, 6]

        density_fig, ax = plt.subplots(figsize=(fig_specs[0], fig_specs[1]))
        cs = ax.scatter(x, y, **kwargs_CMD)
        plt.gca().invert_yaxis()

        if cluster_obj:
            plt.title(cluster_obj.name.replace("_", " "))
            ax.set_xlabel(cluster_obj.CMD_specs["axes"][1])
            ax.set_ylabel(f"absolute {cluster_obj.CMD_specs['axes'][0]}")
        elif title_axes_specs:
            plt.title(title_axes_specs[0])
            ax.set_xlabel(title_axes_specs[1])
            ax.set_ylabel(title_axes_specs[2])
        else:
            ax.set_xlabel(f"color index")
            ax.set_ylabel(f"absolute magnitude")
        plt.colorbar(cs, ax=ax)

        return density_fig
    elif kde:
        return kde_A
    else:
        return x, y, kwargs_CMD

if __name__ == "__main__":

    from Classfile import *
    from pre_processing import cluster_df_list, cluster_name_list, WD_filter, CIII_clusters_new, CIII_df
    import my_utility
    from Empirical_iso_reader import merged_BPRP
    # 0.1 Set the correct output paths
    output_path = my_utility.set_output_path()

    # 0.3 Create the archive from all the loaded data files
    Archive_clusters = np.concatenate(cluster_name_list, axis=0)
    Archive_df = pd.concat(cluster_df_list, axis=0)

    sns.set_style("darkgrid")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 16
    save_plot = True

    # ----------------------------------------------------------------------------------------------------------------------
    MG_comp = merged_BPRP[merged_BPRP["Cluster_id"].isin(["Stock_2", "RSG_8", "Ruprecht_147", "NGC_3532"])]

    for n, cluster in enumerate(Archive_clusters):
        if cluster in ["Stock_2", "RSG_8", "Ruprecht_147", "NGC_3532"]:

            # 1. Create a class object for each cluster
            df = Archive_df[Archive_df["Cluster_id"] == cluster]
            OC = star_cluster(cluster, df)
            OC.create_CMD()


            CMD_density_design(OC.CMD, cluster_obj=OC, density_plot=False)
            #OC.kwargs_CMD["s"] = 40

            fig_talk, ax = plt.subplots(1, 1, figsize=(3.5, 5))

            plt.subplots_adjust(left=0.17, bottom=0.12, top=0.99, right=0.975, wspace=0.0)

            ax.scatter(OC.density_x, OC.density_y, **OC.kwargs_CMD)
            ax.set_xlabel(r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$")
            ax.set_ylabel(r"$\mathrm{M}_{\mathrm{G}}$", labelpad=1)

            result_df = MG_comp[MG_comp["Cluster_id"] == OC.name]

            plt.plot(result_df["l_x"], result_df["l_y"], color="grey", label="5. perc")
            plt.plot(result_df["m_x"], result_df["m_y"], color="red", label="Isochrone")
            plt.plot(result_df["u_x"], result_df["u_y"], color="grey", label="95. perc")

            ylim = ax.get_ylim()

            ax.set_ylim(ylim[1], ylim[0])
            ax.text(0.2,9.5, s =OC.name.replace("_"," "))

            fig_talk.show()

            if save_plot:
                fig_talk.savefig(output_path + "CMD_{}.pdf".format(OC.name), dpi=500)