# Bayesian Optimization attempts for European XFEL tuning

__Basic Idea__: It reads the configuration from a dict or a json / yaml file and starts the Bayesian optimization.

## Additional Notes

- Currently a basic plotting is implemented in `utils.plot_progress`, but using it in VS-Code shows some flickering (see this [issue](https://github.com/microsoft/vscode/issues/132143)), in plain jupyter notebook it works smoothly.
- `pydoocs` require additional package `jpeg` that needs to be installed via conda `conda install -c conda-forge jpeg`.

## Problem

The json configuration files contain the following tuning tasks

- `SASE1_CAX_CAY.json` SASE1 tuning with undulator air coil correctors
- `SASE1_matching_quads.json` SASE1 tuning with quadrupole magnets
- `SASE2_CAX_CAY.json`, `SASE2_CAX_CAY_Launch.json` SASE2 tuning with air coil correctors
- `SASE2_matching_quads.json` SASE2 tuning with quadrupole magnets

## Beamtime Planning

- __Task 0__: get the environment working and fix any possible bug
- Define a (_not optimized_, _reproducible_) initial working condition
  - Save initial condition
  - Set back to this initial condition every time for fair comparison

### Task 1: SASE1 tuning with air coils

1. Apply BO for ~~first~~ _randomly selected_ 4 correctors, find proper settings
   1. Test BO with UCB, EI
   2. Test BO with different __proximal step sizes__
      1. 3 different proximal lengthscales from short to long
      2. __hard vs. proximal__ step size limits
   3. Test BO with __default settings__ vs. __explicit measured noise__
2. Increase the number of input parameters
   1. to 6 air coils
   2. to 8 air coils
   3. increase until failure?
3. Benchmark the BO performance against simplex (maybe for 6 parameters)

### Task 2: SASE1 tuning with Quadrupoles (optional to do task 3 first)

1. Use previously found BO settings, apply for `./conf/SASE1_matching_quads.json`

### Task 3: SASE2 tuning with air coils

If SASE2 is available

1. Use previously found BO settings, repeat step 2 in __Task 1__

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

---

## Tasks

TODO:

- [x] Basic BO functionality
  - [x] BO Loop
  - [x] Basic logging
- [ ] Advanced control:
  - [x] Custom start condition: random initialization, start from current setting
  - [ ] Fine tuning the Acq, Prior...?
  - [x] Step size control: hard / proximal biasing
  - [ ] More information in the logging
- [ ] Preliminary Tests:
  - [x] BO test with simple mathematical functions
  - [x] I/O test with pydoocs
  - [x] Environment test on xfeluser server
  - [ ] (Check compatibility) Environment test on control room PCs
  - [ ] Provide a beam time note book with the defined procedures
- [ ] Other features
  - [x] live progress plot
- [ ] (If enough time): try direct interface with [Xopt](https://github.com/ChristopherMayes/Xopt), this will be a sustainable and more preferred way for productive tools.
