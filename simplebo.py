import time
from typing import Callable, Optional, Union

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class SimpleBO:
    def __init__(
        self,
        problem_config: dict,
        readonly=False,
        active_params: Optional[list(int)] = None,
        acquisition: str = "EI",
        outcome_transofrm: Optional[OutcomeTransform] = Standardize,
        **kwargs
    ) -> None:
        self.read_config(problem_config, active_params)
        self.acquisition = acquisition
        self.readonly = readonly

        self.outcome_transform = outcome_transofrm
        # Initialize Logging
        self.steps_taken = 0

        pass

    def read_config(self, config_file: dict, active_param: Optional[list] = None):

        if active_param is None:  # use all channels listed
            active_param = list(range(config_file["id"]))
        self.input_params = config_file["id"][active_param]
        self.n_params = len(self.input_params)
        self.objective_func = config_file["fun_a"]
        self.max_steps = config_file["max_iter"]
        self.bounds = torch.tensor(config_file["lims"])[active_param]

        pass

    def optimize(self, callback: Optional[Callable] = None):

        # Optimization Loop
        while self.steps_taken < self.max_steps:
            self.step()
            if callback is not None:  # Do something between steps?
                callback

    def init_bo(self):
        # initial design of the BO, ...
        pass

    def step(self):
        """Take one BO step"""

        x_next = self.suggest_next_sample()

        y = self.evaluate_objective(x_next.detach().numpy())

        # Log history
        self.steps_taken += 1  # increase step count
        # Append data
        self.X = torch.cat([self.X, x_next])
        self.Y = torch.cat([self.Y, y])
        pass

    def suggest_next_sample(self) -> torch.Tensor:

        # Construct and fit GP
        self.gp = SingleTaskGP(self.X, self.Y, outcome_transform=self.outcome_transform)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        # Build acquisition function
        self.acqf = self._build_acqf()
        # Maximize acquisition
        candidates, _ = optimize_acqf(
            acq_function=self.acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"maxiter": 200},
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

    def evaluate_objective(self, input) -> torch.Tensor:
        # Set new parameters
        if not self.readonly:
            _set_new_parameters(input, param_names=self.input_params)
        else:
            print("Testing: will skip setting parameters...")
        # Get objective function
        objective = _get_objective(obj_func=self.objective_func)
        return torch.Tensor(objective)

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
    pass


def _get_objective(
    obj_func: str, nreadings: int = 1, interval: float = 0.1, minimize=False
) -> Union[float, np.ndarray]:

    assert nreadings > 0
    objs = None
    for _ in range(nreadings):
        objs.append()
        time.sleep(interval)  # old fashioned way :)
    averaged_obj = np.mean(objs)
    if minimize:  # invert the objective if minimization
        averaged_obj *= -1
    return averaged_obj
