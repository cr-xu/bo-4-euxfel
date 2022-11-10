import botorch
import gpytorch
import torch
import numpy as np
from typing import Optional, Union


def simple_bayesopt(problem_config: dict, **kwargs) -> dict:
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
    log = {}
    max_steps = problem_config["max_iter"]
    if kwargs.has_key("active_param"):  # use selected channels as input
        active_param = kwargs["active_param"]
    else:  # use all channels listed
        active_param = list(range(problem_config["id"]))
    input_params = problem_config["id"][active_param]

    # BO Loop
    for i in range(max_steps):
        # Get Next Parameter Setting
        new_action = suggest_next_sample()
        # Set New Parameters
        new_action = new_action.detach().numpy()
        set_new_parameters(input_params, new_action)
        # Evaluate Function

        # Log the optimization step

    # Optional, set back to best value

    return log


def suggest_next_sample() -> torch.Tensor:

    # Construct and Fit GP

    # Build Acquisition

    # Maximize Acquisition

    # Return parameter setting for next evaluation
    pass


def set_new_parameters(input: np.ndarray, param_names: Optional[list[str]]=None, **kwargs):
    # set new parameters via pydoocs
    pass
