from __future__ import annotations

from typing import Any

import pandas as pd

from semeio.fmudesign.utils import map_dependencies


class Sensitivity:
    sensvalues: pd.DataFrame

    def __init__(self, sensname: str, verbosity: int = 0) -> None:
        """
        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
            verbosity (int): How much information to print. Non-negative integer.
        """
        self.sensname: str = sensname
        self.verbosity: int = verbosity

    def map_dependencies(self, dependencies: dict[str, Any]) -> Sensitivity:
        """Map the dependencies, mutating the dataframe `self.sensvalues`."""
        verbose = self.verbosity > 0  # Because the function takes a boolean
        self.sensvalues: pd.DataFrame = map_dependencies(
            self.sensvalues, dependencies=dependencies, verbose=verbose
        )
        return self
