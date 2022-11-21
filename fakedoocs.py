"""A dummy module providing pyDOOCS similar interface for testing purpose"""
from typing import Union

import numpy as np


class DOOCSTestFunction:

    test_output_channels = ["Ackley", "Rosenbrock"]

    def __init__(self) -> None:
        self.x1 = 0
        self.x2 = 0

    def write(self, channel, value) -> bool:
        if channel == "test/variable/x1":
            self.x1 = value
        elif channel == "test/variable/x2":
            self.x2 = value
        else:
            raise NotImplementedError(
                f'Writing channel "{channel}" is not implemented yet '
            )
        return True

    def read(self, channel) -> dict:
        if channel not in self.test_output_channels:
            raise NotImplementedError(
                f'Reading channel "{channel}" is not implemented yet '
            )
        if channel == "Rosenbrock":
            value = rosenbrock(np.array([self.x1, self.x2]))
        elif channel == "Ackley":
            value = ackley(np.array([self.x1, self.x2]))
        elif channel == "test/variable/x1":
            value = self.x1
        elif channel == "test/variable/x2":
            value = self.x2

        return {
            "data": value,
        }


dummy = DOOCSTestFunction()


def read(channel) -> dict:
    return dummy.read(channel)


def write(channel, value):
    return dummy.write(channel, value)


# Some 2D test functions, c.f. https://en.wikipedia.org/wiki/Test_functions_for_optimization
def ackley(x: np.ndarray) -> Union[float, np.ndarray]:
    y = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    y -= np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    y += np.e + 20
    return -y


def rosenbrock(x: np.ndarray) -> Union[float, np.ndarray]:
    y = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    return -y
