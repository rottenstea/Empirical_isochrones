.. toctree::
   :maxdepth: 2


Extraction
==========

``star_cluster class``
-----------------------

Input 
   Cluster identifier, photometry and parallax observations

A ``star_cluster`` instance is created from the input data. For this object, various methods are available for performing the various pre-processing and computation steps
necessary for the full pipeline of the isochrone extraction. In short, the workflow is as follows:


1. Compute a Color-Magnitude diagram and transform it using Principal Component Analysis ``star_cluster.create_CMD`` (quick 'n' dirty version available as well).
2. Create a weight-map from the observation uncertainties ``star_cluster.create_weights``.
3. Tune hyperparameters for Support Vector regression and save the best results ``star_cluster.SVR_Hyperparameter_tuning`` and ``gridsearch_and_ranking``.
4. In case the hyperparameters have already been determined: ``star_cluster.SVR_read_from_file``.
5. Extract a single empirical curve ``star_cluster.curve_extraction``.
6. Resample a large number of curves from bootstrapped cluster data ``star_cluster.resample_curve``.
7. Calculate the median and uncertainty bounds ``star_cluster.interval_stats``.

The last three steps can be called simuntaneously using the ``star_cluster.isochrone_and_intervals`` method.


.. autoclass:: EmpiricalArchive.Extraction.Classfile.star_cluster
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

Further ``star_cluster`` functions
-----------------------------------

Subsidiary functions called by the ``star_cluster`` class object.

.. automodule:: EmpiricalArchive.Extraction.Classfile
   :members: abs_mag_error, RSS
   :undoc-members:
   :show-inheritance:


Empirical isochrone reader
--------------------------

Function for reading in the empirical isochrones saved for each cluster and for creating a master-table containing
all results for the empirical Archive.


.. automodule:: EmpiricalArchive.Extraction.Empirical_iso_reader
   :members: build_empirical_df
   :undoc-members:
   :show-inheritance:


Pre-processing
--------------

Function that converts the raw .csv input files into pd.DataFrames and only retains the necessary
columns. It is automatically called to do this transformation for 8 different catalogs in the main script, so
that they may just be imported into the various python scripts by their variable name or the variable name of the list.


.. automodule:: EmpiricalArchive.Extraction.pre_processing
   :members: create_df, create_reference_csv
   :undoc-members:
   :show-inheritance: