# Bayesian Optimization attempts for European XFEL tuning

__Basic Idea__: It reads the configuration from a dict or a json / yaml file and starts the Bayesian optimization.

## Project Structure

- `simplebo.py` Implements the basic BO logic

## Usage

### Installing Requirements

It is recommended to use a virtual environment, e.g. conda. With the environment activated, simply do

```bash
pip install -r requirements.txt
```

### Using BO

---

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
- [ ] (If enough time): try direct interface with [Xopt](https://github.com/ChristopherMayes/Xopt), this will be a sustainable and more preferred way for productive tools.
