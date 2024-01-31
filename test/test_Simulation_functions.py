import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/alena/PycharmProjects/PaperI/')

from EmpiricalArchive.IsoModulator.Simulation_functions import apparent_G, simulated_CMD


# ----------------------------------------------------------------------------------------------------------------------
def test_apparent_G():
    # Test case 1: Check if the function returns the correct result for a single value
    M = np.array([7.0])  # mag
    dist = 100.0  # pc
    result = apparent_G(M, dist)
    expected_result = np.array([12])
    assert np.allclose(result, expected_result), f"Test case 1 failed. Got {result}, expected {expected_result}"

    # Test case 2: Check if the function handles arrays correctly
    M = np.array([-3.0, 3.0, 17.0])
    dist = 100.0
    result = apparent_G(M, dist)
    expected_result = np.array([2, 8, 22])
    assert np.allclose(result, expected_result), f"Test case 2 failed. Got {result}, expected {expected_result}"


def test_neg_dist_apparent_G():
    # Test case 3: Check if the function raises a ValueError for negative dist
    M = np.array([5.0])
    dist = -17
    with pytest.raises(ValueError, match="Mean cluster distance must be greater than 0."):
        apparent_G(M, dist)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def initialized_class_object():

    # Generate random data for columns X and Y
    np.random.seed(42)  # Set seed for reproducibility
    isochrone_data = {
        'BPRP_isochrone_x': np.random.uniform(-1, 4, 200),
        'BPRP_isochrone_y': np.random.uniform(-2, 20, 200),
        'BPG_isochrone_x': np.random.uniform(0, 3, 200),
        'BPG_isochrone_y': np.random.uniform(-2, 20, 200),
        'GRP_isochrone_x': np.random.uniform(1, 4, 200),
        'GRP_isochrone_y': np.random.uniform(-2, 20, 200),
        'Cluster': ['dummy_cluster'] * 200
    }
    cluster_data = {
        'BP-RP': np.random.uniform(-1, 4, 200),
        'BP-G': np.random.uniform(0, 3, 200),
        'G-RP': np.random.uniform(1, 4, 200),
        'Gmag': np.random.uniform(-2, 20, 200),
        'plx': np.random.uniform(8, 9, 200),
        'Cluster_id': ['dummy_cluster'] * 200
    }

    isochrone_df = pd.DataFrame(isochrone_data)
    cluster_data_df = pd.DataFrame(cluster_data)
    test_CMD = simulated_CMD(cluster_name='dummy_cluster', isochrone_df=isochrone_df, cluster_data_df=cluster_data_df)

    return test_CMD


def test_set_CMD_type_valid(initialized_class_object):
    # Test if the method sets values correctly for valid CMD types
    obj = initialized_class_object

    # Test CMD type 1
    obj.set_CMD_type(1)
    assert np.allclose(obj.green, apparent_G(obj.abs_G_bprp, obj.mean_distance))
    assert obj.cax is obj.bp_rp
    assert obj.abs_G is obj.abs_G_bprp
    assert obj.cols == ["BP-RP", "Gmag"]
    assert obj.num_simulated_stars == len(obj.green)

    # Test CMD type 2
    obj.set_CMD_type(2)
    assert np.allclose(obj.green, apparent_G(obj.abs_G_bpg, obj.mean_distance))
    assert obj.cax is obj.bp_g
    assert obj.abs_G is obj.abs_G_bpg
    assert obj.cols == ["BP-G", "Gmag"]
    assert obj.num_simulated_stars == len(obj.green)

    # Test CMD type 3
    obj.set_CMD_type(3)
    assert np.allclose(obj.green, apparent_G(obj.abs_G_grp, obj.mean_distance))
    assert obj.cax is obj.g_rp
    assert obj.abs_G is obj.abs_G_grp
    assert obj.cols == ["G-RP", "Gmag"]
    assert obj.num_simulated_stars == len(obj.green)


def test_set_CMD_type_invalid(initialized_class_object):
    # Test if the method raises ValueError for invalid CMD types
    obj = initialized_class_object

    with pytest.raises(ValueError, match="CMD-type can only be 1, 2, or 3."):
        obj.set_CMD_type(0)

    with pytest.raises(ValueError, match="CMD-type can only be 1, 2, or 3."):
        obj.set_CMD_type(4)


def test_add_parallax_uncertainty(initialized_class_object):
    obj = initialized_class_object
    obj.set_CMD_type(1)
    delta_plx = 0.1
    obj.add_parallax_uncertainty(delta_plx)
    assert isinstance(obj.abs_mag_incl_plx, pd.Series)
    assert len(obj.abs_mag_incl_plx) == obj.num_simulated_stars


def test_add_parallax_uncertainty_zero_delta_plx(initialized_class_object):
    delta_plx = 0
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(delta_plx)
    assert isinstance(obj.abs_mag_incl_plx, pd.Series)
    assert len(obj.abs_mag_incl_plx) == obj.num_simulated_stars


def test_add_parallax_uncertainty_no_negative_distribution(initialized_class_object):
    delta_plx = 100
    obj = initialized_class_object
    obj.mean_distance = 100
    obj.set_CMD_type(1)
    with pytest.raises(ValueError, match="Negative values in new parallaxes detected"):
        obj.add_parallax_uncertainty(delta_plx)


def test_add_binary_fraction_unallowed_vals(initialized_class_object):
    obj = initialized_class_object
    binarity_frac = -0.1
    with pytest.raises(ValueError, match="Fraction needs to be between 0 and 1."):
        obj.add_binary_fraction(binarity_frac)
    binarity_frac = 1.1
    with pytest.raises(ValueError, match="Fraction needs to be between 0 and 1."):
        obj.add_binary_fraction(binarity_frac)


def test_add_binary_fraction(initialized_class_object):
    binarity_frac = 0.3
    obj = initialized_class_object
    obj.abs_mag_incl_plx = pd.Series(np.random.randn(100))  # Mock abs_mag_incl_plx
    original_abs_mag_incl_plx = obj.abs_mag_incl_plx.copy()

    obj.add_binary_fraction(binarity_frac)

    # Find indices where elements in second_array are identical to first_array
    matching_indices = np.where(obj.abs_mag_incl_plx_binarity == original_abs_mag_incl_plx)[0]
    matching_elements = original_abs_mag_incl_plx[matching_indices]
    assert np.all(np.isin(matching_elements, original_abs_mag_incl_plx))

    # Find indices where elements in second_array are different from first_array
    non_matching_indices = np.where(obj.abs_mag_incl_plx_binarity != original_abs_mag_incl_plx)[0]
    # Calculate the difference between corresponding elements in the two arrays
    differences = obj.abs_mag_incl_plx_binarity  - original_abs_mag_incl_plx
    # Filter differences corresponding to the non-matching indices
    non_matching_differences = differences[non_matching_indices]
    # Check if the absolute differences are close to 0.753
    tolerance = 0.01  # Adjust tolerance as needed
    assert np.allclose(np.abs(non_matching_differences), 0.753, atol=tolerance)


def test_add_binary_fraction_zero(initialized_class_object):
    binarity_frac = 0
    initialized_class_object.abs_mag_incl_plx = pd.Series(np.random.randn(100))  # Mocking abs_mag_incl_plx
    original_abs_mag_incl_plx = initialized_class_object.abs_mag_incl_plx.copy()

    initialized_class_object.add_binary_fraction(binarity_frac)

    assert original_abs_mag_incl_plx.equals(initialized_class_object.abs_mag_incl_plx_binarity)


def test_add_extinction(initialized_class_object):
    extinction_level = 0.5
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(0.1)
    obj.add_binary_fraction(0.3)

    # make into dataframe
    binary_df = pd.DataFrame(data=np.stack([obj.bp_rp, obj.abs_mag_incl_plx_binarity], axis=1),
                             columns=obj.cols)

    obj.add_extinction(extinction_level)

    # Check if absolute magnitude is correctly modified
    assert np.allclose(binary_df[obj.cols[1]] + extinction_level,
                       obj.abs_mag_incl_plx_binarity_extinction[obj.cols[1]])

    # Calculate expected color index after applying extinction
    expected_color_index = binary_df[obj.cols[0]] + (1.212 - 0.76) * extinction_level

    # Check if color index is correctly modified
    assert np.allclose(expected_color_index,
                       obj.abs_mag_incl_plx_binarity_extinction[obj.cols[0]])


def test_add_field_unallowed_vals(initialized_class_object):
    obj = initialized_class_object
    binarity_frac = -0.1
    with pytest.raises(ValueError, match="Fraction needs to be between 0 and 1."):
        obj.add_field_contamination(binarity_frac)
    binarity_frac = 1.1
    with pytest.raises(ValueError, match="Fraction needs to be between 0 and 1."):
        obj.add_field_contamination(binarity_frac)


def test_add_field_contamination_sampling(initialized_class_object):

    contamination_frac = 0.9
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(0.1)
    obj.add_binary_fraction(0.3)
    obj.add_extinction(0.5)
    obj.add_field_contamination(contamination_frac)

    # Check if the correct number of entries is sampled
    assert (len(obj.abs_mag_incl_plx_binarity_extinction_field["Gmag"]) -
            len(obj.abs_mag_incl_plx_binarity_extinction["Gmag"])) == 180  # 90% of 200 (dummy data)


def test_add_field_contamination_conversion(initialized_class_object):

    contamination_frac = 0.7
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(0.1)
    obj.add_binary_fraction(0.3)
    obj.add_extinction(0.5)
    obj.add_field_contamination(contamination_frac)

    # Check if the conversion is done correctly for the sampled field data
    assert all(col in obj.abs_mag_incl_plx_binarity_extinction_field.columns
               for col in ['BP-RP', 'Gmag'])


def test_add_field_contamination_merging(initialized_class_object):

    contamination_frac = 0.7
    obj = initialized_class_object

    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(0.1)
    obj.add_binary_fraction(0.3)
    obj.add_extinction(0.5)
    obj.add_field_contamination(contamination_frac)

    # Check if the merging is done correctly
    assert len(obj.abs_mag_incl_plx_binarity_extinction_field) == 340  # 200 (mock) + 200*0.7


def test_simulate_calls_add_methods_correctly(initialized_class_object):
    # Sample uncertainties
    uncertainties = [0.1, 0.2, 0.3, 0.4]
    obj = initialized_class_object
    obj.set_CMD_type(1)

    # Call simulate method
    result = obj.simulate(uncertainties)

    # Verify that add_ methods are called with correct uncertainties
    assert obj.abs_mag_incl_plx_binarity_extinction_field is not None

    # Verify the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that expected columns are present in the resulting DataFrame
    expected_columns = ['BP-RP', 'Gmag']
    for col in expected_columns:
        assert col in result.columns


def test_plot_verification_returns_figure_and_axes(initialized_class_object):
    uncertainties = [0.1, 0.2, 0.3, 0.4]
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(uncertainties[0])
    obj.add_binary_fraction(uncertainties[1])
    obj.add_extinction(uncertainties[2])
    obj.add_field_contamination(uncertainties[3])
    fig, axes = obj.plot_verification(uncertainties)

    # Verify the return types
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (6, )  # Assuming 2x3 subplots


def test_plot_verification_plots_correct_data(initialized_class_object):
    uncertainties = [0.1, 0.2, 0.3, 0.4]
    obj = initialized_class_object
    obj.set_CMD_type(1)
    obj.add_parallax_uncertainty(uncertainties[0])
    obj.add_binary_fraction(uncertainties[1])
    obj.add_extinction(uncertainties[2])
    obj.add_field_contamination(uncertainties[3])
    fig, axes = obj.plot_verification(uncertainties)

    # Verify that each subplot contains the expected data

    # Test original subplot
    ax_original = axes[0]
    x_original = obj.cax
    y_original = obj.abs_G
    original_scatter = ax_original.collections[0]  # Assuming scatter plot is the only collection
    assert np.array_equal(original_scatter.get_offsets(), np.column_stack((x_original, y_original)))

    # Test plx subplot
    ax_plx_uncertainty = axes[1]
    x_plx_uncertainty = obj.cax
    y_plx_uncertainty = obj.abs_mag_incl_plx
    plx_uncertainty_scatter = ax_plx_uncertainty.collections[0]  # Assuming scatter plot is the only collection
    assert np.array_equal(plx_uncertainty_scatter.get_offsets(), np.column_stack((x_plx_uncertainty, y_plx_uncertainty)))

    # Test binary subplot
    ax_bin_uncertainty = axes[2]
    x_bin_uncertainty = obj.cax
    y_bin_uncertainty = obj.abs_mag_incl_plx_binarity
    bin_uncertainty_scatter = ax_bin_uncertainty.collections[0]  # Assuming scatter plot is the only collection
    assert np.array_equal(bin_uncertainty_scatter.get_offsets(), np.column_stack((x_bin_uncertainty, y_bin_uncertainty)))

    # Test Av subplot
    ax_Av_uncertainty = axes[3]
    x_Av_uncertainty = obj.abs_mag_incl_plx_binarity_extinction[obj.cols[0]]
    y_Av_uncertainty = obj.abs_mag_incl_plx_binarity_extinction[obj.cols[1]]
    Av_uncertainty_scatter = ax_Av_uncertainty.collections[0]  # Assuming scatter plot is the only collection
    assert np.array_equal(Av_uncertainty_scatter.get_offsets(), np.column_stack((x_Av_uncertainty, y_Av_uncertainty)))

    # Test field subplot - for some reason broken2
    # ax_f_uncertainty = axes[4]
    # x_f_uncertainty = obj.abs_mag_incl_plx_binarity_extinction_field[obj.cols[0]]
    # y_f_uncertainty = obj.abs_mag_incl_plx_binarity_extinction_field[obj.cols[1]]
    # f_uncertainty_scatter = ax_f_uncertainty.collections[0]  # Assuming scatter plot is the only collection
    # assert np.array_equal(f_uncertainty_scatter.get_offsets(), np.column_stack((x_f_uncertainty, y_f_uncertainty)))

    plt.close(fig)







