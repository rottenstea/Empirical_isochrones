import numpy as np
import pytest
import pandas as pd

import sys

sys.path.append('/Users/alena/PycharmProjects/PaperI/')

from EmpiricalArchive.IsoModulator.Simulation_functions import apparent_G, simulated_CMD


# ----------------------------------------------------------------------------------------------------------------------
def test_apparent_G():
    # Test case 1: Check if the function returns the correct result for a single value
    M = np.array([7.0])  # mag
    dist = 100.0  # pc
    result = apparent_G(M, dist)
    expected_result = np.array([12])  # Replace this with the expected result for the given inputs
    assert np.allclose(result, expected_result), f"Test case 1 failed. Got {result}, expected {expected_result}"

    # Test case 2: Check if the function handles arrays correctly
    M = np.array([-3.0, 3.0, 17.0])
    dist = 100.0
    result = apparent_G(M, dist)
    expected_result = np.array([2, 8, 22])  # Replace this with the expected result for the given inputs
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

# Run the tests by executing 'pytest' in your terminal
