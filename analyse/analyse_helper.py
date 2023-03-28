import json

import numpy as np


def load_ocelot_data(fname):
    with open(fname, "r") as f:
        ocelot_data = json.load(f)
    actuator_names = [
        key for key in ocelot_data.keys() if key.split("/")[0] == "XFEL.FEL"
    ]
    xvalues = np.array([ocelot_data[key] for key in actuator_names]).T
    loaded_data = {
        "metadata": {
            "input": actuator_names,
            "objective": ocelot_data["function"].split()[-1],
            "nreadings": ocelot_data["nreadings"][0],
        },
        "X": xvalues.tolist(),
        "Y": ocelot_data["obj_values"],
        "Y_std": ocelot_data["std"],
    }
    return loaded_data
