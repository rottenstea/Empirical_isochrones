import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from plotly import graph_objects as go
from plotly.offline import plot
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.stats import gaussian_kde


def CMD_density_design(A, gamma: float = 0.7, to_RBG: list = None, from_RBG: list = None, marker: str = ".",
                       s: int = 50, density_plot=True, kde: bool = False, cluster_obj=None,
                       fig_specs: list = None, title_axes_specs: list = None):
    """
     Density map and plots of the clusters.

    :param A: 2D array of scatter data to be color-coded according to its density
    :param gamma: Power-norm coefficient
    :param to_RBG: Lighter color
    :param from_RBG: Darker color
    :param marker: Marker-Style
    :param s: Marker-Size
    :param density_plot: returns figure if true
    :param kde: return kde
    :param cluster_obj: cluster object instance for grabbing names and labels
    :param fig_specs: Figure size
    :param title_axes_specs: list of title, x-axis, y-axis labels
    :return: either figure, kde, or x_density, y_density + plotting kwargs
    """
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

    kwargs_CMD = dict(c=z, cmap=cluster_map, marker=marker, s=s, lw=0, norm=PowerNorm(gamma))

    if density_plot:
        sns.set_style("darkgrid")
        if not fig_specs:
            fig_specs = [4, 6]

        density_fig, fig_ax = plt.subplots(figsize=(fig_specs[0], fig_specs[1]))
        cs = fig_ax.scatter(x, y, **kwargs_CMD)
        plt.gca().invert_yaxis()

        if cluster_obj:
            plt.title(cluster_obj.name.replace("_", " "))
            fig_ax.set_xlabel(cluster_obj.CMD_specs["axes"][1])
            fig_ax.set_ylabel(f"absolute {cluster_obj.CMD_specs['axes'][0]}")
        elif title_axes_specs:
            plt.title(title_axes_specs[0])
            fig_ax.set_xlabel(title_axes_specs[1])
            fig_ax.set_ylabel(title_axes_specs[2])
        else:
            fig_ax.set_xlabel(f"color index")
            fig_ax.set_ylabel(f"absolute magnitude")
        plt.colorbar(cs, ax=fig_ax)

        return density_fig
    elif kde:
        return kde_A
    else:
        return x, y, kwargs_CMD


def masterplot(isochrone_data: list, age_col: str = "ref_age", id_col: str = "Cluster_id",
               interactive_plot: bool = False,
               age_limits: list = None, plotting_dict: dict = None, one_panel: int = None,
               save_plot_path: str = None):
    """
    Function for plotting the most important summary plot of all empirical isochrones calculated for the archive.

    :param isochrone_data: List of calculated empirical isochrones.
    :param age_col: Column name for the reference age data.
    :param id_col: Column name for the cluster names.
    :param interactive_plot: Boolean indicator for whether an interactive (html) version of the figure should be created
    :param age_limits: Upper and lower reference age limits of isochrones that should be plotted.
    :param plotting_dict: Dictionary with specific plotting parameters.
    :param one_panel: Only plot the BP-RP CMD.
    :param save_plot_path: Path to output location.
    :return: Figure
    """

    if one_panel is None:
        main_data = isochrone_data[0]
    else:
        main_data = isochrone_data[one_panel]

    # design colorbar and norm
    a = plt.get_cmap("YlGnBu_r")
    norm = plt.Normalize(main_data[age_col].min(), main_data[age_col].max())
    sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
    sm.set_array([])
    dict1 = {'hue': age_col, 'palette': a, 'hue_norm': norm}

    if plotting_dict is None:
        plotting_dict_full = {'x': 'm_x', 'y': 'm_y', 'legend': False, 'sort': False, 'lw': 1,
                              'units': id_col, 'estimator': None} | dict1
    else:
        plotting_dict_full = plotting_dict | dict1

    if not age_limits:
        age_low = 5.5
        age_high = 11
    else:
        age_low, age_high = age_limits

    age_info = [age_col, age_high, age_low]

    if not interactive_plot:

        if one_panel is None:
            figure = Archive_plot_2D(isochrone_data=isochrone_data, age_info=age_info, plotting_dict=plotting_dict_full,
                                     scalar_mappable=sm)
        else:
            figure = Archive_plot_2D(isochrone_data=[main_data], age_info=age_info, plotting_dict=plotting_dict_full,
                                     scalar_mappable=sm, one_panel=True)

        if save_plot_path:
            figure.savefig(save_plot_path + "2D_summary.pdf", dpi=600)

    else:
        sorted_clusters = main_data.drop_duplicates(subset=id_col)
        cm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm).to_rgba(sorted_clusters[age_col], alpha=None,
                                                                       bytes=False, norm=True)
        col_hex = [colors.rgb2hex(c) for c in cm]
        colo_hex = col_hex[:]

        if one_panel is None:
            line_data = []
            for iso_data in isochrone_data:
                line_data.append(
                    [iso_data[iso_data[id_col] == cluster][plotting_dict["x"]] for cluster in sorted_clusters])
                line_data.append(
                    [iso_data[iso_data[id_col] == cluster][plotting_dict["y"]] for cluster in sorted_clusters])
            figure = Archive_plot_interactive(isochrone_data=isochrone_data, line_data=line_data,
                                              cluster_list=sorted_clusters, plotting_dict=plotting_dict_full,
                                              color_list=colo_hex)
        else:
            line_data = [
                [main_data[main_data[id_col] == cluster][plotting_dict["x"]] for cluster in sorted_clusters],
                [main_data[main_data[id_col] == cluster][plotting_dict["y"]] for cluster in sorted_clusters]
            ]
            figure = Archive_plot_interactive(isochrone_data=[main_data], line_data=line_data,
                                              cluster_list=sorted_clusters, plotting_dict=plotting_dict_full,
                                              color_list=colo_hex)

        if save_plot_path:
            plot(figure, filename=save_plot_path + 'Interactive_summary.html')

    return figure


def Archive_plot_2D(isochrone_data: list, age_info: list, plotting_dict: dict, scalar_mappable,
                    one_panel: bool = False):
    """
    Two-dimensional version of the masterplot.

    :param isochrone_data: List of empirical isochrones
    :param age_info: List containing the name of the reference age column, as well as its boundaries.
    :param plotting_dict: Dictionary containing specific plotting parameters.
    :param scalar_mappable: Colormap object for the colorbar
    :param one_panel: Only plot the BP-RP CMD.
    :return: Figure
    """
    age_col, age_low, age_high = age_info

    print(age_col)

    if not one_panel:

        BPRP_data, BPG_data, GRP_data = isochrone_data

        # 2D Plot
        fig_2D, ax = plt.subplots(1, 3, figsize=(7.24551, 4.4), sharey="row")
        plt.subplots_adjust(left=0.08, bottom=0.0, top=0.99, right=0.99, wspace=0.1)

        # Subplot 1
        sns.lineplot(data=BPRP_data[(BPRP_data[age_col] >= age_low) & (BPRP_data[age_col] <= age_high)], ax=ax[0],
                     hue=age_col, **plotting_dict).set(
            xlabel=r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}_{\mathrm{RP}}$ (mag)",
            ylabel=r"$\mathrm{M}_{\mathrm{G}}$ (mag)")
        ax[0].set_xlim(-1.9, 5)
        ax[0].set_ylim(17, -3.5)

        # Subplot 2
        sns.lineplot(data=BPG_data[(BPG_data[age_col] >= age_low) & (BPG_data[age_col] <= age_high)], ax=ax[1],
                     hue=age_col, **plotting_dict).set(xlabel=r"$\mathrm{G}_{\mathrm{BP}} - \mathrm{G}$ (mag)")

        ax[1].set_xlim(-1.1, 3.2)
        ax[1].set_ylim(17, -4.1)

        # Subplot 3
        sns.lineplot(data=GRP_data[(GRP_data[age_col] >= age_low) & (GRP_data[age_col] <= age_high)], ax=ax[2],
                     hue=age_col, **plotting_dict).set(xlabel=r"$\mathrm{G}- \mathrm{G}_{\mathrm{RP}}$ (mag)")

        ax[2].set_xlim(-0.5, 1.8)
        ax[2].set_ylim(17, -4.2)

        # Set colorbar
        c = fig_2D.colorbar(scalar_mappable, ax=[ax[0], ax[1], ax[2]], location='bottom', fraction=0.1, aspect=35,
                            pad=0.12)
        cax = c.ax
        cax.tick_params(labelsize=10)
        cax.text(6.65, 0.3, 'log age')

        return fig_2D

    else:

        data_1panel = isochrone_data[0]

        fig_poster, ax = plt.subplots(1, 1, figsize=(3.5, 4.4))
        plt.subplots_adjust(left=0.16, bottom=0.11, top=0.98, right=0.97, wspace=0.0)

        sns.lineplot(data=data_1panel, ax=ax, **plotting_dict).set(ylabel=r"$\mathrm{M}_{\mathrm{G}}$")
        # ax.set_xlim(-0.5, 4.1)
        ax.set_ylim(14, -3.5)
        fig_poster.colorbar(scalar_mappable, ax=ax, location='right', fraction=0.15, aspect=20, pad=0.05)
        plt.text(5.2, 18, 'log age')

        return fig_poster


def Archive_plot_interactive(isochrone_data, line_data: list, cluster_list: list, plotting_dict, color_list,
                             one_panel: bool = False):
    """
    Interactive plot of the empirical isochrones in the archive.

    :param isochrone_data: Cluster data
    :param line_data: List containing all empirical isochrone data
    :param cluster_list: Namelist of the clusters
    :param plotting_dict: Dictionary containing specific plotting parameters.
    :param color_list: Colors for the isochrones
    :param one_panel: Only plot the BP-RP CMD.
    :return: Interactive figure (Html-file)
    """
    if not one_panel:

        BPRP_data, BPG_data, GRP_data = isochrone_data
        BPRP_x, BPRP_y, BPG_x, BPG_y, GRP_x, GRP_y = line_data

        cluster_id = plotting_dict["units"]

        # Initialize figure
        fig = go.Figure()
        fig.update_xaxes(range=[-1, 4])
        fig.update_yaxes(range=[15, -4])
        fig.update_layout(width=750, height=850)
        fig.update_layout(template='plotly_white')

        # Customize legend labels
        cluster_labels = []
        for i, cluster in enumerate(cluster_list):
            if "_" in cluster:
                c_label = "%s: %s Myr" % (cluster.replace("_", " "), round(10 ** (
                    BPRP_data[BPRP_data[cluster_id] == cluster][
                        "ref_age"].unique()[0]) / 1e6, 2))
            else:
                c_label = "%s: %s Myr" % (cluster, round(10 ** (
                    BPRP_data[BPRP_data[cluster_id] == cluster][
                        "ref_age"].unique()[0]) / 1e6, 2))

            unified_label = f"{c_label:_<25}"
            cluster_labels.append(unified_label)

            trace = go.Scatter(x=BPRP_data[BPRP_data[cluster_id] == cluster][plotting_dict['x']],
                               y=BPRP_data[BPRP_data[cluster_id] == cluster][plotting_dict['y']],
                               name=unified_label,
                               visible=True,  # make the first line visible by default
                               line=dict(color=color_list[i]))
            fig.add_trace(trace)

        # create button layout
        updatemenus = [
            dict(
                type='buttons',
                x=0.75,
                y=-0.05,
                direction="right",
                showactive=True,
                buttons=[
                    dict(
                        label=r'BP-RP',
                        method='update',
                        args=[{'x': BPRP_x, 'y': BPRP_y, 'name': cluster_labels}, {"xaxis.range": [-2, 4]}]),
                    dict(
                        label='BP-G',
                        method='update',
                        args=[{'x': BPG_x, 'y': BPG_y, 'name': cluster_labels}, {"xaxis.ramge": [-1, 3]}]),
                    dict(
                        label='G-RP',
                        method='update',
                        args=[{'x': GRP_x, 'y': GRP_y, 'name': cluster_labels}, {"xaxis.ramge": [0, 3]}])
                ]
            )
        ]

        # update layout with button layout
        fig.update_layout(updatemenus=updatemenus)

        # Add slider
        steps = []

        for i, cluster in enumerate(cluster_list):
            df = BPRP_data[BPRP_data["Cluster_id"] == cluster]
            df = df.sort_values("m_y")

            visible_traces = [j <= i + 1 for j in range(len(cluster_list))]
            step = dict(
                method="update",
                args=[{"visible": visible_traces}],
                label=df["ref_age"].unique()[0])
            steps.append(step)

        sliders = [dict(
            active=0,
            y=-0.02,
            currentvalue={"prefix": "log age: "},
            pad={"t": 50},
            steps=steps,
        )]

        # update layout with slider
        fig.update_layout(
            sliders=sliders,
            yaxis_title="absolute G magnitude",
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
            title={
                'text': 'Empirical isochrones',
                'x': 0.5,
                'xanchor': 'center'
            }
        )

        fig.update_layout(
            plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
            xaxis=dict(gridcolor='white'),  # Set the x-axis grid line color to white
            yaxis=dict(gridcolor='white'),  # Set the y-axis grid line color to white
        )

    else:

        main_data = isochrone_data[0]
        m_x, m_y = line_data

        cluster_id = plotting_dict["units"]

        # Initialize figure
        fig = go.Figure()
        fig.update_xaxes(range=[-1, 4])
        fig.update_yaxes(range=[15, -4])
        fig.update_layout(width=750, height=850)
        fig.update_layout(template='plotly_white')

        # Customize legend labels
        cluster_labels = []
        for i, cluster in enumerate(cluster_list):
            if "_" in cluster:
                c_label = "%s: %s Myr" % (cluster.replace("_", " "), round(10 ** (
                    main_data[main_data[cluster_id] == cluster][
                        "ref_age"].unique()[0]) / 1e6, 2))
            else:
                c_label = "%s: %s Myr" % (cluster, round(10 ** (
                    main_data[main_data[cluster_id] == cluster][
                        "ref_age"].unique()[0]) / 1e6, 2))

            unified_label = f"{c_label:_<25}"
            cluster_labels.append(unified_label)

            trace = go.Scatter(x=main_data[main_data[cluster_id] == cluster][plotting_dict['x']],
                               y=main_data[main_data[cluster_id] == cluster][plotting_dict['y']],
                               name=unified_label,
                               visible=True,  # make the first line visible by default
                               line=dict(color=color_list[i]))
            fig.add_trace(trace)

        # create button layout
        updatemenus = [
            dict(
                type='buttons',
                x=0.75,
                y=-0.05,
                direction="right",
                showactive=True,
                buttons=[
                    dict(
                        label=r'BP-RP',
                        method='update',
                        args=[{'x': m_x, 'y': m_y, 'name': cluster_labels}, {"xaxis.range": [-2, 4]}]),
                ]
            )
        ]

        # update layout with button layout
        fig.update_layout(updatemenus=updatemenus)

        # Add slider
        steps = []

        for i, cluster in enumerate(cluster_list):
            df = main_data[main_data["Cluster_id"] == cluster]
            df = df.sort_values("m_y")

            visible_traces = [j <= i + 1 for j in range(len(cluster_list))]
            step = dict(
                method="update",
                args=[{"visible": visible_traces}],
                label=df["ref_age"].unique()[0])
            steps.append(step)

        sliders = [dict(
            active=0,
            y=-0.02,
            currentvalue={"prefix": "log age: "},
            pad={"t": 50},
            steps=steps,
        )]

        # update layout with slider
        fig.update_layout(
            sliders=sliders,
            yaxis_title="absolute G magnitude",
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
            title={
                'text': 'Empirical isochrones',
                'x': 0.5,
                'xanchor': 'center'
            }
        )

        fig.update_layout(
            plot_bgcolor='rgb(234, 234, 241)',  # Set the background color to gray
            xaxis=dict(gridcolor='white'),  # Set the x-axis grid line color to white
            yaxis=dict(gridcolor='white'),  # Set the y-axis grid line color to white
        )

    return fig
