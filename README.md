# IsoModulator

## Description

IsoModulator is a Python package designed as a sub-routine for the larger coding endeavor known as the Empirical Isochrone Archive. The Empirical Isochrone Archive provides empirical isochrones for over 80 nearby open clusters and allows users to calculate their own empirical isochrones for stellar clusters or other agglomerates in the solar neighborhood (d < 500 pc) based on their Color-Magnitude diagrams (CMDs). 

The primary purpose of IsoModulator is to quantify the influence of various astrophysical parameters on the final extraction result. Specifically, the impact of changes to the following variables is studied:

- Parallax
- Unresolved binary fraction
- Extinction level
- Field contamination fraction

These parameters play a crucial role in the accurate determination of empirical isochrones and understanding their variations is essential for precise astrophysical analyses.

## Intended Users

IsoModulator is intended for anyone working with photometric data of stellar clusters or groups and is interested in determining an empirical isochrone. The package can be particularly useful for:

- Researchers and astronomers working on stellar cluster studies.
- Individuals interested in relative age determination methods through the comparison of empirical isochrones.
- Users seeking quality control and calibration measures for fitting theoretical isochrones with various existing packages.

## Installation

You can install IsoModulator locally using the `pyproject.toml` file. Open your terminal at the top-level project directory and run:

```bash
python3 -m pip install .
```

In the near future, the package will be uploaded to PyPi and available via 

```bash
pip install IsoModulator
```

## Getting Started

To get started with IsoModulator, import the module in your Python script or Jupyter notebook:

```python
import EmpiricalArchive.IsoModulator
```
The main part of the code is included in the class `simulated_CMD`, that can be found in the `IsoModulator.Simulation_functions`
package.

## Documentation

The documentation for the project is provided via `Sphinx` and can be found [here](docs/build/html/index.html).

## License

This project is licensed under the [MIT License](LICENSE).
