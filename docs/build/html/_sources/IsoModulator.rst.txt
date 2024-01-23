.. toctree::
   :maxdepth: 2

IsoModulator
============

The ``IsoModulator`` module contains the ``Simulated_CMD`` class, as well as further supplementary functions. The ``Simulated_CMD`` class is
allows users to provide a sample empirical isochrone as well as uncertainty values for the parameters:

1. parallax
2. unresolved binary fraction
3. field contamination fraction
4. extinction level

Based on the input values and the position of the provided isochrone, a CMD is simulated and a new isochrone is calculated.

Simulated_CMD class
-------------------

Input
   Cluster name or identifier, empirical isochrone data, cluster CMD data

Lelele.

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
