# Bayesian Optimization attempts for European XFEL tuning

__Basic Idea__: It reads the configuration from a dict or a json / yaml file and starts the Bayesian optimization.

## Project Structure

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
- [ ] (If enough time): try direct interface with [Xopt](https://github.com/ChristopherMayes/Xopt), this will be a sustainable and more preferred way for productive tools.
