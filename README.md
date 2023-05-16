# Bayesian Optimization attempts for European XFEL tuning

It reads the configuration from a `dict` or a json / yaml file and starts the Bayesian optimization.

__Important notes for new BO Practitioner__: The code in this repository serves as a good initial point to understand how BO can be implemented. For a more systematic and maintainable solution, we kindly refer to the [Xopt]([Practitioner__](https://github.com/ChristopherMayes/Xopt)) package, which provides more BO variants and is designed to be a generic optimizer.

The results are published in the IPAC'23 proceeding: [https://doi.org/10.18429/JACoW-IPAC-23-THPL028](https://doi.org/10.18429/JACoW-IPAC-23-THPL028) ([Pre-proceeding version](https://www.ipac23.org/preproc/doi/thpl028/index.html))

## Additional Notes

- Currently a basic plotting is implemented in `utils.plot_progress`, but using it in VS-Code shows some flickering (see this [issue](https://github.com/microsoft/vscode/issues/132143)), in plain jupyter notebook it works smoothly.
- `pydoocs` require additional package `jpeg` that needs to be installed via conda `conda install -c conda-forge jpeg`.

## Problem

The json configuration files contain the following tuning tasks

- `SASE1_CAX_CAY.json` SASE1 tuning with undulator air coil correctors
- `SASE1_matching_quads.json` SASE1 tuning with quadrupole magnets
- `SASE2_CAX_CAY.json`, `SASE2_CAX_CAY_Launch.json` SASE2 tuning with air coil correctors
- `SASE2_matching_quads.json` SASE2 tuning with quadrupole magnets

---

## Repository Structure

- `simplebo.py` Implements the basic BO logic
- `conf/*.json` Configuration files for optimization tasks
  - `test_rosenbrock.json` A template for optimizating a test function
- `utils.py` Utility functions, e.g. the proximial acquisition function [implemented in Xopt](https://github.com/ChristopherMayes/Xopt/blob/main/xopt/generators/bayesian/custom_botorch/proximal.py)

## Usage

### Installing Requirements

It is recommended to use a virtual environment, e.g. `conda`. With the environment activated, simply do

```bash
pip install -r requirements.txt
```

### Using BO

(_For pydoocs users_): Simply define your problem in a json file structured as the Ocelot optimizer configuration. Load the config into the `SimpleBO` object and start with `optimize`.

---

## Tasks

TODO:

- [x] Basic BO functionality
  - [x] BO Loop
  - [x] Basic logging
- [x] Advanced control:
  - [x] Custom start condition: random initialization, start from current setting
  - [x] Fine tuning the Acq, Prior...
  - [x] Step size control: hard / proximal biasing
  - [x] More information in the logging
- [x] Preliminary Tests:
  - [x] BO test with simple mathematical functions
  - [x] I/O test with pydoocs
  - [x] Environment test on xfeluser server
  - [x] (Check compatibility) Environment test on control room PCs
  - [x] Provide a beam time note book with the defined procedures
- [ ] Other features
  - [x] live progress plot
- [ ] (If enough time): try direct interface with [Xopt](https://github.com/ChristopherMayes/Xopt), this will be a sustainable and more preferred way for productive tools.
