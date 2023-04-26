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
