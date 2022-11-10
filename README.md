# Bayesian Optimization attempts for European XFEL tuning

Basic Idea: It reads the configuration from a dict or a json / yaml file and starts the Bayesian optimization.

## Project Structure

- `simplebo.py` Implements the basic BO logic

## Tasks

TODO:

- [ ] Basic BO functionality
  - [ ] BO Loop
  - [ ] Meaningful logging
- [ ] Advanced control:
  - [ ] Custom start condition: random initialization, start from current setting
  - [ ] fine tuning the Acq, Prior
  - [ ] Step size control: hard / proximal biasing
- [ ] Preliminary Tests:
  - [ ] BO test with simple mathematical functions
  - [ ] I/O test with pydoocs
  - [ ] (Check compatibility) Environment test on control room PCs
- [ ] (If enough time): try direct interface with [Xopt](https://github.com/ChristopherMayes/Xopt)
