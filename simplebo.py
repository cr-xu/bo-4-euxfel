import json
import pickle
import time
from datetime import datetime
from typing import Callable, Optional, Union

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import fakedoocs as pydoocs
from utils import ProximalAcquisitionFunction

# import pydoocs


class SimpleBO:
    def __init__(
        self,
        problem_config: dict,
        readonly: bool = False,
        active_params: Optional[list[int]] = None,
        n_init: int = 5,
        acquisition: str = "EI",
        fixed_noise: bool = False,
        outcome_transofrm: Optional[OutcomeTransform] = Standardize,
        input_transform: Optional[InputTransform] = Normalize,
        step_limit_type: str = "proximal",
        step_size: Union[
            float, np.ndarray
        ] = 0.1,  # as percent of the bound, if local initialization or use hard step size limit
        proximal_len: Union[float, np.ndarray] = 0.5,  # if proximal step size limit
        logfile: str = "default_bolog",
    ) -> None:
        self.read_config(problem_config, active_params)
        self.acquisition = acquisition
        self.readonly = readonly
        self.n_init = n_init
        self.fixed_noise = fixed_noise

        self.outcome_transform = outcome_transofrm(1)  # standardize y
        self.input_transform = input_transform(self.n_params)  # normalize x

        self.step_size = torch.tensor(step_size)
        if isinstance(proximal_len, torch.Tensor):
            self.proximal_len = proximal_len
        else:
            self.proximal_len = torch.ones(self.n_params) * proximal_len
        self.step_limit_type = step_limit_type
        self.logfilename = logfile

        # Initialize Logging
        self.reset()
        pass

    def read_config(self, config_file: dict, active_param: Optional[list] = None):
        """Load settings from a configuration dict

        Parameters
        ----------
        config_file : dict
            Dictionary containing at least the following keys:
            `["id","lims","max_iter","nreadings","interval","fun_a"]`
        active_param : Optional[list], optional
            Indices of the parameter to be used, if `None`, will use all
            the param provided in the `config_file["id"]`
        """
        if active_param is None:  # use all channels listed
            active_param = np.array(range(len(config_file["id"])))
        self.input_params = [config_file["id"][p] for p in active_param]
        self.n_params = len(self.input_params)
        self.objective_func = config_file["fun_a"]
        self.max_iter = config_file["max_iter"]
        self.nreadings = config_file["nreadings"]
        self.interval = config_file["interval"]
        self.bounds = torch.tensor(config_file["lims"])[active_param].T
        if "maximization" in config_file:
            self.maximize = config_file["maximization"]
        else:
            self.maximize = True
        pass

    def optimize(
        self,
        init_mode: str = "random",
        callback: Optional[Callable] = None,
        save_log: bool = True,
        fname: Optional[str] = None,
    ):
        """A wrapper to start full optimization

        It first initialize GP with random settings, then run the bo until
        `max_iter` is reached, writes the history to some log file

        Parameters
        ----------
        callback : Optional[Callable], optional
            A custom callback function between each step, by default None
        save_log : bool, optional
            Whether to save history to log file at the end, by default True
        fname : Optional[str], optional
            Name of the log file, by default None
        """
        if not self.initialized:
            self.init_bo(mode=init_mode)

        # Optimization Loop
        while self.steps_taken < self.max_iter:
            self.step()
            if callback is not None:  # Do something between steps?
                callback()

        if save_log:
            self.save(filename=fname)

    def reset(self):
        """Resets to initial state, clean history, similar idea as a gym `env.reset()`"""
        self.history = {
            "metadata": {},
            "X": [],
            "Y": [],
            "Y_std": [],
            "Initial": {},
        }
        self._update_metadata_in_history()
        self.X = None
        self.Y = None
        self.Y_std = None
        self.steps_taken = 0
        self.initialized = False
        self.initial_settings = self._get_current_setting()

    def _update_metadata_in_history(self):
        new_dict = {
            "input": self.input_params,
            "objective": self.objective_func,
            "nreadings": self.nreadings,
            "interval": self.interval,
            "max_iter": self.max_iter,
            "bounds": self.bounds.detach().tolist(),
            "acquisition": self.acquisition,
            "step_limit_type": self.step_limit_type,
            "proximal_len": self.proximal_len.tolist(),
            "step_size": self.step_size.tolist(),
            "fixed_noise": self.fixed_noise,
        }
        self.history["metadata"].update(new_dict)

    def init_bo(self, mode="current"):
        # initial design of the BO, sample random initial points
        if (self.X is not None) or (self.Y is not None):
            print("Warning: BO already initialized before, resetting...")
            self.reset()

        if mode == "current":  # initialize locally around current settings
            init_settings = torch.tesnor(self.initial_settings)
            self.X = init_settings + self.step_size * (
                torch.rand((self.n_init, self.n_params)).double()
                * (self.bounds[1] - self.bounds[0])
                + self.bounds[0]
            )
            # make sure the candidates are in limit
            self.X = torch.clamp(self.X, min=self.bounds[0], max=self.bounds[1])
        elif mode == "random":  #
            self.X = (
                torch.rand((self.n_init, self.n_params)).double()
                * (self.bounds[1] - self.bounds[0])
                + self.bounds[0]
            )
        self.Y = torch.zeros(self.n_init, 1).double()
        self.Y_std = torch.zeros(self.n_init, 1).double()

        # Sample initial settings
        for i, x in enumerate(self.X):
            y, std = self.evaluate_objective(x.detach().numpy())
            self.Y[i] = y
            self.Y_std[i] = std

        self.initialized = True

        # Append initial samples
        self.history["initial"] = {
            "X": self.X.detach().tolist(),
            "Y": self.Y.detach().tolist(),
            "Y_std": self.Y_std.detach().tolist(),
        }

    def step(self):
        """Take one BO step"""

        x_next = self.suggest_next_sample()
        y, std = self.evaluate_objective(x_next.detach().numpy().squeeze())

        # Append data
        self.X = torch.cat([self.X, x_next])
        self.Y = torch.cat([self.Y, y])
        self.Y_std = torch.cat([self.Y_std, std])

        # Log history
        self.steps_taken += 1  # increase step count
        self.history["X"] = self.X.detach().tolist()
        self.history["Y"] = self.Y.detach().flatten().tolist()
        self.history["Y_std"] = self.Y_std.detach().flatten().tolist()

    def suggest_next_sample(self) -> torch.Tensor:
        """Core BO step, suggest next candidate setting"""
        # Construct and fit GP
        if self.fixed_noise:
            self.gp = FixedNoiseGP(
                train_X=self.X,
                train_Y=self.Y,
                train_Yvar=self.Y_std**2,
                outcome_transform=self.outcome_transform,
                input_transform=self.input_transform,
            )
        else:
            self.gp = SingleTaskGP(
                train_X=self.X,
                train_Y=self.Y,
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

        # Constrain step size
        if self.step_limit_type == "hard":
            # calculate new bound
            allowed_action_size = (
                self.bounds[1] - self.bounds[1]
            ) * self.step_size  # convert to float32
            newbounds = [
                self.X[-1] - allowed_action_size,
                self.X[-1] + allowed_action_size,
            ]
            candidates = torch.clamp(candidates, min=newbounds[0], max=newbounds[1])

        # Return parameter setting for next evaluation
        return candidates

    def _build_acqf(self):
        """Construct Acquisition function"""
        if self.acquisition == "EI":
            self.acqf = ExpectedImprovement(self.gp, self.Y.max())
        elif self.acquisition == "UCB":
            self.acqf = UpperConfidenceBound(self.gp, beta=0.2)
        elif self.acquisition == "PI":
            self.acqf = ProbabilityOfImprovement(self.gp, self.Y.max())
        if self.step_limit_type == "proximal":
            # Apply proximal biasing
            # print("Proximal biasing to be implemented...")
            self.acqf = ProximalAcquisitionFunction(
                self.acqf, proximal_weights=self.proximal_len
            )
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
        objective, std = _get_objective(
            obj_func=self.objective_func,
            nreadings=self.nreadings,
            interval=self.interval,
            maximize=self.maximize,
        )
        return torch.tensor([[objective]]), torch.tensor([[std]])

    def _get_current_setting(self) -> list:
        p_current = []
        for param in self.input_params:
            p_current.append(pydoocs.read(channel=param)["data"])
        return p_current

    def save(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"log/bo_log_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=4)

    def save_to_pkl(self, filename: Optional[str] = None):

        if filename is None:
            filename = f"log/bo_log_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.history, f)


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
) -> tuple:
    """Read the objective channel value

    Parameters
    ----------
    obj_func : str
        DOOCS address of the optimization objective
    nreadings : int, optional
        Number of obs to be averaged, by default 1
    interval : float, optional
        Time to wait inbetween, by default 0.1
    maximize : bool, optional
        Whether it's a maximizaiton problem, by default True

    Returns
    -------
    Union[float, np.ndarray]
        (Averaged) objective value
    """
    assert nreadings > 0
    objs = []
    for _ in range(nreadings):
        objs.append(pydoocs.read(obj_func)["data"])
        time.sleep(interval)  # old fashioned way :)

    obj_mean, obj_std = np.mean(objs), np.std(objs)

    if not maximize:  # invert the objective if minimization
        obj_mean *= -1

    return obj_mean, obj_std
