.. toctree::
   :maxdepth: 2

IsoModulator
============

The ``IsoModulator`` module contains the ``simulated_CMD`` class, as well as further supplementary functions. The ``simulated_CMD`` class is
allows users to provide a sample empirical isochrone as well as uncertainty values for the parameters:

1. parallax
2. unresolved binary fraction
3. field contamination fraction
4. extinction level

Based on the input values and the position of the provided isochrone, a CMD is simulated and a new isochrone is calculated. 

``simulated_CMD class``
-----------------------

Input
   Cluster name or identifier, empirical isochrone, cluster CMD data

Using the input data, a class object is generated. It consists of *N_clustermembers* synthetic stars placed along the originally calculated empirical isochrone, which
are assumed to reside at the mean cluster distance. After determining the ``simulated_CMD.CMD_type()``, where

- 1: Gaia *BP-RP vs. absG*
- 2: Gaia *BP-G vs. absG*
- 3: Gaia *G-RP vs. absG*,

the parameter uncertainties can be added iteratively to the synthetic star data. Using the method ``simulated_CMD.simulate()``, a new isochrone is computed using the
normal ``Extraction`` routine, and can be compared to the original one using various metrics.

.. autoclass:: EmpiricalArchive.IsoModulator.Simulation_functions.simulated_CMD
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

Further functions
-----------------

Subsidiary functions called by the ``Simulated_CMD`` class object.

.. automodule:: EmpiricalArchive.IsoModulator.Simulation_functions
   :members: apparent_G
   :undoc-members:
   :show-inheritance:
