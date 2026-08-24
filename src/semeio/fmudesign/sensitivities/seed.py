from collections.abc import Sequence
from typing import Any

import pandas as pd

from .sensitivity import Sensitivity


class SeedSensitivity(Sensitivity):
    """
    A seed sensitivity is normally the reference for one by one sensitivities,
    which all other sensitivities are compared to. All parameters will be at
    their default values. Only the RMS_SEED will be varying.

    It contains a list of seeds to be repeated for each sensitivity
    The parameter name is hardcoded to RMS_SEED
    It will be assigned the sensname 'p10_p90' which will be written to
    the SENSCASE column in the output.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
        seedname: str,
        seedvalues: Sequence[int] | None,
        parameters: dict[str, Any] | None,
    ) -> None:
        """Generates parameter values for a seed sensitivity

        Args:
            size (int): number of rows to generate
            seedname (str): name of seed parameter to add
            seedvalues (list): list of integer seedvalues
            parameters (dict): parameter names and
                distributions or values.
        """
        if seedvalues is None:
            msg = (
                "Seed values must be set when running sensitivity type 'seed'. "
                f"Got seed: {seedvalues}"
            )
            raise ValueError(msg)

        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues[seedname] = seedvalues[0:size]

        if parameters is not None:
            for key in parameters:
                dist_name = parameters[key][0].lower()
                constant = parameters[key][1]
                if dist_name != "const":
                    raise ValueError(
                        'A sensitivity of type "seed" can only have '
                        "additional parameters where dist_name is "
                        f'"const". Check sensitivity {self.sensname}"'
                    )
                self.sensvalues[key] = constant

        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"
