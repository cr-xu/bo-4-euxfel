import botorch
import gpytorch
import numpy as np
import torch
import time
from typing import Optional, Union


def simple_bayesopt(problem_config: dict, readonly=False, **kwargs) -> dict:
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
    n_params = len(input_params)
    objective_func = problem_config["fun_a"]

    # Initialize the BO
    X = torch.zeros([1, n_params])
    Y = torch.zeros([1, 1])

    # BO Loop
    for i in range(max_steps):
        # Get Next Parameter Setting
        new_action = suggest_next_sample()
        new_action = new_action.detach().numpy()  # convert to numpy

        # Evaluate Function
        y = evaluate_objective(
            new_action, objective_func=objective_func, readonly=readonly
        )

        # Append data

        # Log the optimization step

    # Optional, set back to best value

    return log


def suggest_next_sample() -> torch.Tensor:

    # Construct and Fit GP

    # Build Acquisition

    # Maximize Acquisition

    # Return parameter setting for next evaluation
    pass


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


def evaluate_objective(
    input: np.ndarray,
    param_names: Optional[list[str]] = None,
    objective_func: str = None,
    readonly: bool = False,
    **kwargs
) -> torch.Tensor:
    # Set new parameters
    if not readonly:
        _set_new_parameters(input, param_names=param_names)
    else:
        print("Testing: will skip setting parameters...")
    # Get objective function
    objective = _get_objective(obj_func=objective_func)
    return torch.Tensor(objective)
