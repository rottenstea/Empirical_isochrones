.. toctree::
   :maxdepth: 2
   :caption: Contents:

   My_tools.rst
   Extraction.rst
   IsoModulator.rst
   Examples.md


Welcome to EmpiricalArchive's Documentation!
============================================

This is the main documentation for the EmpiricalArchive for generating purely observation-based isochrones for nearby open clusters.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

``my_tools`` Module
===================

Collection of small utility and plotting functions vital for core functionality of the code.

.. automodule:: EmpiricalArchive.My_tools
   :members:
   :undoc-members:
   :show-inheritance:

``Extraction`` Module
=======================

Centerpiece of the code, responsible for the generation of empirical isochrones, from pre-processing of the observational data, extraction and bootstrapping of 
isochrones in every desired Color-magnitude combination, to storing the results and creating summary tables.

.. automodule:: EmpiricalArchive.Extraction
   :members:
   :undoc-members:
   :show-inheritance:

``Isomodulator`` Module
=======================

Module for computing the influence of various parameter uncertainties (edge cases) on the shape of the cluster CMD distribution and therefore on the reported empirical isochrones.
It mainly serves as a quantification for the reliability of the observation-based isochronal curves.

.. automodule:: EmpiricalArchive.IsoModulator
   :members:
   :undoc-members:
   :show-inheritance:

:Author(s):
    Alena Rottensteiner

:Version: 1.0.0 of 2024/01/20
