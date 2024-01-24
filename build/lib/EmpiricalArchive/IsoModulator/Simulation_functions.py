import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from EmpiricalArchive.Extraction.Classfile import star_cluster


def apparent_G(M: np.array, dist: float) -> np.array:
    """
    Function for calculating the apparent magnitude from the absolute magnitude and mean cluster distance.

    :param M: Absolute magnitude
    :param dist: Mean cluster distance, scalar value
    :return: array of the apparent magnitudes
    """
    if dist < 0:
        raise ValueError("Mean cluster distance must be greater than 0.")

    if len(M) == 0:
        return np.array([])

    d = np.empty(shape=len(M))
    d.fill(dist)
    return 5 * np.log10(d) - 5 + M


class simulated_CMD:
    def __init__(self, cluster_name: str, isochrone_df: pd.DataFrame, cluster_data_df: pd.DataFrame,
                 dataset_id: str = "1"):
        """
        ClassObject for simulating a Color-magnitude diagram (CMD) for a given cluster, based on an existing empirical
        isochrone of the cluster. The original cluster data is required as input as well, as the mean cluster distance
        is computed for the simulated stars to transform color index and absolute magnitudes into apparent magnitudes.

        :param cluster_name: String indicating the cluster name or label listed in the "Cluster" column of the isochrone
        dataframe.
        :param isochrone_df: Dataframe comprising the original empirical isochrones, either provided by the archive or
        by the user.
        :param cluster_data_df: Dataframe containing the astrometric and photometric data used for creating the initial
        isochrone.
        """

        # initialize other stuff
        self.green = None
        self.cax = None
        self.abs_G = None
        self.cols = None
        self.num_simulated_stars = None
        self.abs_mag_incl_plx = None
        self.abs_mag_incl_plx_binarity = None
        self.abs_mag_incl_plx_binarity_extinction = None
        self.abs_mag_incl_plx_binarity_extinction_field = None

        # choose cluster
        self.name = cluster_name
        self.cluster_data = isochrone_df[isochrone_df["Cluster"] == cluster_name]

        # set CMD variables
        self.abs_G_bprp = self.cluster_data["BPRP_isochrone_y"]
        self.bp_rp = self.cluster_data["BPRP_isochrone_x"]

        self.abs_G_bpg = self.cluster_data["BPG_isochrone_y"]
        self.bp_g = self.cluster_data["BPG_isochrone_x"]

        self.abs_G_grp = self.cluster_data["GRP_isochrone_y"]
        self.g_rp = self.cluster_data["GRP_isochrone_x"]

        # define cluster object (for distances)
        OC = star_cluster(cluster_name, cluster_data_df, dataset_id="1")
        self.mean_distance = float(np.mean(OC.distance))

    def set_CMD_type(self, CMD_type: int):
        """
        Class method for determining the type of CMD that will be created. Currently set to the Gaia CMD options.
        1) BP-RP vs absolute G
        2) BP-G vs absolute G
        3) G-RP vs absolute G

        :param CMD_type: 1,2 or 3
        :return: None
        """
        # calculate apparent G mag
        if CMD_type == 1:
            self.green = apparent_G(self.abs_G_bprp, self.mean_distance)
            self.cax = self.bp_rp
            self.abs_G = self.abs_G_bprp
            self.cols = ["BP-RP", "Gmag"]
        elif CMD_type == 2:
            self.green = apparent_G(self.abs_G_bpg, self.mean_distance)
            self.cax = self.bp_g
            self.abs_G = self.abs_G_bpg
            self.cols = ["BP-G", "Gmag"]
        elif CMD_type == 3:
            self.green = apparent_G(self.abs_G_grp, self.mean_distance)
            self.cax = self.g_rp
            self.abs_G = self.abs_G_grp
            self.cols = ["G-RP", "Gmag"]

        self.num_simulated_stars = len(self.green)

    def add_parallax_uncertainty(self, delta_plx: float):
        """
        Method for adding parallax uncertainty to the simulated CMD. For each star in the simulated CMD, the original
        parallax is combined with a relative uncertainty drawn from a normal distribution that has its extrema
        (99.7 % coverage)nat +/- delta parallax. In the last step, the absolute magnitude is calculated again.

        :param delta_plx: Uncertainty fraction of the parallax.
        :return: None
        """

        lower_bound = -delta_plx
        upper_bound = delta_plx

        # Calculate mean and standard deviation based on the desired bounds
        mean = (upper_bound + lower_bound) / 2
        std_dev = (upper_bound - lower_bound) / 6  # Dividing by 6 for 99.7% coverage within the range

        # Generate a normal distribution
        normal_distribution = np.random.normal(mean, std_dev, self.num_simulated_stars)

        # convert mean distance to plx
        plx = 1000 / self.mean_distance

        # add the parallax uncertainties sampled from the normal distribution bounded by delta plx
        new_dist = 1000 / (plx + plx * normal_distribution)

        # print(self.mean_distance, np.mean(new_dist), np.std(new_dist), max(new_dist), min(new_dist))

        # transform to absolute mag again
        self.abs_mag_incl_plx = self.green - 5 * np.log10(new_dist) + 5

    def add_binary_fraction(self, binarity_frac: float):
        """
        Method for adding an artificial unresolved binary fraction to the simulated CMD. For a random fraction of the
        simulated stars, defined by the parameter binary_frac, the absolute magnitude is increased by -0.753 mag
        (increase for equal mass main sequence unresolved binaries). Recommended use: After adding parallax
        uncertainty.

        :param binarity_frac: Fraction of unresolved binaries [0,1].
        :return: None
        """

        binary_frame = self.abs_mag_incl_plx.copy()
        # Randomly sample 30% of the elements
        sampled_indices = binary_frame.sample(frac=binarity_frac).index

        # Apply the shift to the original series based on sampled indices
        binary_frame.loc[sampled_indices] -= 0.753

        self.abs_mag_incl_plx_binarity = binary_frame

    def add_extinction(self, extinction_level: float):
        """
        Method for adding a constant extinction level to the CMD data in both absolute magnitude and color index. In
        case of the absolute magnitudes, the extinction level is directly added. In case of the color index, the color
        excess is calculated by first multiplying the Gaia DR3 extinction coefficients (approximated with Draine+2003,
        R_V = 3.1 and the assumption of a flat SED) with the extinction level, and then subtracting them.
        Recommended use: After adding parallax uncertainty and binary fraction.

        :param extinction_level: in Gaia G-band magnitudes (1 AV = 0.789 AG)
        :return: None
        """

        # make into dataframe
        cluster_df = pd.DataFrame(data=np.stack([self.bp_rp, self.abs_mag_incl_plx_binarity], axis=1),
                                  columns=self.cols)

        # calculate extinction in the various passbands and excess factor E
        A_BP = 1.212 * extinction_level
        A_RP = 0.76 * extinction_level
        E = A_BP - A_RP

        cluster_df[self.cols[1]] += extinction_level
        cluster_df[self.cols[0]] += E

        self.abs_mag_incl_plx_binarity_extinction = cluster_df

    def add_field_contamination(self, contamination_frac: float,
                                field_data_path: str =
                                '/Users/alena/PycharmProjects/PaperI/EmpiricalArchive/data/Gaia_DR3/Gaia_DR3_500pc_1percent.csv'):
        """
        Method for adding a specified fraction of field contamination to the cluster CMD from a field data catalog. The
        data are randomly sampled and converted to the CMD format (color index and absolute magnitude). The field data
        are then added to the cluster data.
        Recommended use: After adding parallax uncertainty, binarity fraction and extinction level.

        :param contamination_frac: Fraction of field contamination with respect to the cluster that should be sampled.
        :param field_data_path: Path to the field data to sample from (default: 1% of Gaia DR3 sources within 500 pc).
        :return: None
        """

        # load slimmed catalog
        data = pd.read_csv(field_data_path)

        # Randomly sample contaminated entries
        n = int(self.num_simulated_stars * contamination_frac)
        sampled_data = data.sample(n=n)

        # use only the relevant columns
        field_df = sampled_data[
            ["parallax", "parallax_error", "ruwe", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]]

        # generate CMD columns
        if self.cols[0] == "BP-RP":
            field_df.loc[:, "BP-RP"] = field_df.loc[:, 'phot_bp_mean_mag'] - field_df.loc[:, 'phot_rp_mean_mag']
        elif self.cols[0] == "BP-G":
            field_df.loc[:, "BP-G"] = field_df.loc[:, 'phot_bp_mean_mag'] - field_df.loc[:, 'phot_g_mean_mag']
        elif self.cols[0] == "G-RP":
            field_df.loc[:, "G-RP"] = field_df.loc[:, 'phot_g_mean_mag'] - field_df.loc[:, 'phot_rp_mean_mag']

        field_df.loc[:, self.cols[1]] = field_df.loc[:, 'phot_g_mean_mag'] - 5 * np.log10(
            1000 / field_df.loc[:, "parallax"]) + 5

        # merge with other df
        common_columns = self.abs_mag_incl_plx_binarity_extinction.columns.intersection(field_df.columns)

        # Concatenate based on common columns and reindex
        self.abs_mag_incl_plx_binarity_extinction_field = pd.concat(
            [self.abs_mag_incl_plx_binarity_extinction, field_df[common_columns]], axis=0)

    def simulate(self, uncertainties: list) -> pd.DataFrame:
        """
        Method that automatically adds all four implemented uncertainties (parallax, binary fraction, extinction level,
        field contamination) to the simulated CMD data in the recommended order. A column with the cluster name is also
        added for further processing steps inside the star_cluster class of the main module.

        :param uncertainties: List of uncertainties in the order: parallax, binary fraction, extinction level, field contamination
        :return: DataFrame with the final CMD data and the cluster name
        """
        u_plx, binarity, extinction, field = uncertainties

        self.add_parallax_uncertainty(delta_plx=u_plx)
        self.add_binary_fraction(binarity_frac=binarity)
        self.add_extinction(extinction_level=extinction)
        self.add_field_contamination(contamination_frac=field)

        star_cluster_object = self.abs_mag_incl_plx_binarity_extinction_field
        star_cluster_object["Cluster_id"] = self.name

        return star_cluster_object

    def plot_verification(self, uncertainties: list) -> plt.figure:
        """
        Method for visually presenting the changes to the initially simulated stars along the original empirical
        isochrone. Each subplot shows the cluster CMD after the addition of the respective parameter uncertainty.

        :param uncertainties: List of uncertainties in the order: parallax, binary fraction, extinction level, field contamination
        :return: Figure of the changes brought by the addition of each uncertainty
        """
        fig, ax = plt.subplots(2, 3, figsize=(8, 6))
        plt.subplots_adjust(top=0.95, left=0.05, right=0.98, hspace=0.23, wspace=0.15, bottom=0.05)
        axes = ax.ravel()

        # Plot original
        if self.cols[0] == "BP-RP":
            axes[0].scatter(self.cax, self.abs_G_bprp, s=5, color="blue")
        elif self.cols[0] == "BP-G":
            axes[0].scatter(self.cax, self.abs_G_bpg, s=5, color="blue")
        elif self.cols[0] == "G-RP":
            axes[0].scatter(self.cax, self.abs_G_grp, s=5, color="blue")
        axes[0].set_ylim(15, -4)
        axes[0].legend(loc="best")
        axes[0].set_title("original")

        # plx uncertainty
        axes[1].scatter(self.cax, self.abs_mag_incl_plx, s=5, color="orange", label=uncertainties[0])
        axes[1].set_ylim(15, -4)
        axes[1].legend(loc="best")
        axes[1].set_title("u_plx")

        # binarity
        axes[2].scatter(self.cax, self.abs_mag_incl_plx_binarity, s=5, color="red", label=uncertainties[1])
        axes[2].set_ylim(15, -4)
        axes[2].legend(loc="best")
        axes[2].set_title("binarity")

        # extinction
        axes[3].scatter(self.abs_mag_incl_plx_binarity_extinction[self.cols[0]],
                        self.abs_mag_incl_plx_binarity_extinction[self.cols[1]], s=5, color="green",
                        label=uncertainties[2])
        axes[3].set_ylim(15, -4)
        axes[3].legend(loc="best")
        axes[3].set_title("extinction")

        # contamination
        axes[4].scatter(self.abs_mag_incl_plx_binarity_extinction_field[self.cols[0]],
                        self.abs_mag_incl_plx_binarity_extinction_field[self.cols[1]], s=5, color="violet",
                        label=uncertainties[3])
        axes[4].set_ylim(15, -4)
        axes[4].legend(loc="best")
        axes[4].set_title("field")

        fig.delaxes(axes[5])

        return fig, axes
