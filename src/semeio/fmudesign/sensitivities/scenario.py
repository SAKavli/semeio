from collections.abc import Sequence
from typing import Any

import pandas as pd

from .sensitivity import Sensitivity


class ScenarioSensitivityCase(Sensitivity):
    """Each ScenarioSensitivity can contain one or
    two ScenarioSensitivityCases.

    The 1-2 cases are typically 'low' and 'high' cases for one or
    a set of  parameters, where all realisatons in
    the case have identical values except the seed value
    and in special cases specified background values which may
    vary within the case.

    One or two ScenarioSensitivityCase instances can be added to each
    ScenarioSensitivity object.

    Attributes:
        sensname (str): name of the sensitivity case,
            equals SENSCASE in design matrix.
        sensvalues (pd.DataFrame): parameters and values
            for the sensitivity with realisation numbers as index.

    """

    def generate(
        self,
        size: int,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Generate sensvalues for the ScenarioSensitivityCase

        Args:
            size (int): number of rows to generate
            parameters (dict):
                dictionary with parameter names and values
            seeds (str): default or None
        """

        self.sensvalues = pd.DataFrame(
            columns=list(parameters.keys()), index=range(size)
        )
        for key, value in parameters.items():
            self.sensvalues[key] = value
        self.sensvalues["SENSCASE"] = self.sensname

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[:size]


class ScenarioSensitivity(Sensitivity):
    """Each design can contain one or several single sensitivities of type
    Seed, MonteCarlo or Scenario.
    Each ScenarioSensitivity can contain 1-2 ScenarioSensitivityCases.

    The ScenarioSensitivity class is used for sensitivities where all
    realizatons in a ScenarioSensitivityCase have identical values
    but one or more parameter has a different values from the other
    ScenarioSensitivityCase.

    Exception is the seed value and the special case where
    varying background parameters are specified. Then these are varying
    within the case.

    Attributes:
        case1 (ScenarioSensitivityCase): first case, e.g. 'low case'
        case2 (ScenarioSensitivityCase): second case, e.g. 'high case'
        sensvalues (pd.DataFrame): design values for the sensitivity, containing
           1-2 cases
    """

    case1: ScenarioSensitivityCase | None = None
    case2: ScenarioSensitivityCase | None = None

    def add_case(self, senscase: ScenarioSensitivityCase) -> None:
        """
        Adds a ScenarioSensitivityCase instance
        to a ScenarioSensitivity object.

        Args:
            senscase (ScenarioSensitivityCase):
                Equals SENSCASE in design matrix.
        """
        if self.case1 is not None:  # Case 1 has been read, this is case2
            if senscase.sensvalues is not None and "SENSCASE" in senscase.sensvalues:
                self.case2 = senscase
                senscase.sensvalues["SENSNAME"] = self.sensname
                self.sensvalues = pd.concat(
                    [self.sensvalues, senscase.sensvalues], sort=True
                )
        elif senscase.sensvalues is not None and "SENSCASE" in senscase.sensvalues:
            self.case1 = senscase
            self.sensvalues = senscase.sensvalues.copy()
            self.sensvalues["SENSNAME"] = self.sensname
