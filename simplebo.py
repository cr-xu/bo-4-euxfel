import json
import time
from typing import Callable, Optional, Union

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import fakedoocs as pydoocs

# import pydoocs


class SimpleBO:
    def __init__(
        self,
        problem_config: dict,
        readonly: bool = False,
        active_params: Optional[list[int]] = None,
        acquisition: str = "EI",
        outcome_transofrm: Optional[OutcomeTransform] = Standardize,
        input_transform: Optional[InputTransform] = Normalize,
        step_limit_type: str = "proximal",
        step_size: int = 0.1,
    ) -> None:
        self.read_config(problem_config, active_params)
        self.acquisition = acquisition
        self.readonly = readonly

        self.outcome_transform = outcome_transofrm(1)  # standardize y
        self.input_transform = input_transform(self.n_params)  # normalize x

        self.step_size = step_size
        self.step_limit_type = step_limit_type

        # Initialize Logging
        self.reset()
        pass

    def read_config(self, config_file: dict, active_param: Optional[list] = None):

        if active_param is None:  # use all channels listed
            active_param = np.array(range(len(config_file["id"])))
        self.input_params = [config_file["id"][p] for p in active_param]
        self.n_params = len(self.input_params)
        self.objective_func = config_file["fun_a"]
        self.max_steps = config_file["max_iter"]
        self.nreadings = config_file["nreadings"]
        self.interval = config_file["interval"]
        self.bounds = torch.tensor(config_file["lims"])[active_param].T
        if "maximization" in config_file:
            self.maximize = config_file["maximization"]
        else:
            self.maximize = True
        pass

    def optimize(self, callback: Optional[Callable] = None):
        if not self.initialized:
            self.init_bo()

        # Optimization Loop
        while self.steps_taken < self.max_steps:
            self.step()
            if callback is not None:  # Do something between steps?
                callback()

    def reset(self):
        """Resets to initial state, clean history, similar idea as a gym `env.reset()`"""
        self.history = {"steps:": []}
        self.X = None
        self.Y = None
        self.steps_taken = 0
        self.initialized = False

    def init_bo(self, n_init: int = 5):
        # initial design of the BO, sample random initial points
        if (self.X is not None) or (self.Y is not None):
            print("Warning: BO already initialized before, reinitializing...")

        self.X = torch.rand((n_init, self.n_params)).double() * (self.bounds[1,:] - self.bounds[0,:]) + self.bounds[0,:]
        self.Y = torch.zeros(n_init, 1).double()

        # Sample initial settings
        for i, x in enumerate(self.X):
            y = self.evaluate_objective(x.detach().numpy())
            self.Y[i] = y

        self.initialized = True

    def step(self):
        """Take one BO step"""

        x_next = self.suggest_next_sample()
        y = self.evaluate_objective(x_next.detach().numpy().squeeze())
        # Log history
        self.steps_taken += 1  # increase step count
        # Append data
        self.X = torch.cat([self.X, x_next])
        self.Y = torch.cat([self.Y, y])
        pass

    def suggest_next_sample(self) -> torch.Tensor:
        """Core BO step, suggest next candidate setting"""
        # Construct and fit GP
        self.gp = SingleTaskGP(
            self.X,
            self.Y,
            outcome_transform=self.outcome_transform,
            input_transform=self.input_transform,
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        # Build acquisition function
        self._build_acqf()
        # Calculate new bounds if hard stepsize limit

        # Maximize acquisition
        candidates, _ = optimize_acqf(
            acq_function=self.acqf,
            bounds=self.bounds.float(),  # only works with float type
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"maxiter": 150},
        )

        # Return parameter setting for next evaluation
        return candidates

    def _build_acqf(self):
        if self.acquisition == "EI":
            self.acqf = ExpectedImprovement(self.gp, self.Y.max())
        elif self.acquisition == "UCB":
            self.acqf = UpperConfidenceBound(self.gp, beta=0.2)
        elif self.acquisition == "PI":
            self.acqf = ProbabilityOfImprovement(self.gp, self.Y.max())
        if self.step_limit_type == "proximal":
            # Apply proximal biasing
            print("Proximal biasing to be implemented...")
            pass

    def evaluate_objective(self, input) -> float:
        # Set new parameters
        if isinstance(input, torch.Tensor):
            input = input.detach().numpy()
        if not self.readonly:
            _set_new_parameters(input, param_names=self.input_params)
        else:
            print("Testing: will skip setting parameters...")
        # Get objective function
        objective = _get_objective(
            obj_func=self.objective_func,
            nreadings=self.nreadings,
            interval=self.interval,
            maximize=self.maximize,
        )
        return torch.tensor([[objective]])

    def save(self, filename: str = "log/defaultlog.json"):
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=4)
        pass

    """A simple Bayesian optimization routine

    Parameters
    ----------
    problem_config : dict
        A config ditionary containing the problem definition.
        Something like a `ocelot-optimizer` config file, with the
        following keys:
        `["id","lims","max_iter","nreadings","interval","fun_a"]`

    Returns
    -------
    dict
        Dictionary containing the run history

    Examples
    --------


    """


def _set_new_parameters(
    input: np.ndarray, param_names: Optional[list[str]] = None, **kwargs
):
    # set new parameters via pydoocs
    assert len(input) == len(param_names)

    for i, param in enumerate(param_names):
        pydoocs.write(param, input[i])

    # wait until settle?


def _get_objective(
    obj_func: str, nreadings: int = 1, interval: float = 0.1, maximize: bool = True
) -> Union[float, np.ndarray]:

    assert nreadings > 0
    objs = []
    for _ in range(nreadings):
        objs.append(pydoocs.read(obj_func))
        time.sleep(interval)  # old fashioned way :)

    averaged_obj = np.mean(objs)
    if not maximize:  # invert the objective if minimization
        averaged_obj *= -1

    return averaged_obj
