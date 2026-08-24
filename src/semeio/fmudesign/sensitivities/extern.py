from collections.abc import Sequence

import pandas as pd

from semeio.fmudesign._excel_to_dict import _raise_if_duplicates
from semeio.fmudesign.utils import parameters_from_extern

from .sensitivity import Sensitivity


class ExternSensitivity(Sensitivity):
    """
    Used when reading parameter values from a file
    Assumed to be used with monte carlo type sensitivities and
    will hence write 'p10_p90' as SENSCASE in output designmatrix

    Attributes:
        sensname (str): Name of sensitivity.
            Defines SENSNAME in design matrix
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
        filename: str,
        parameters: list[str],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Reads parameter values for a monte carlo sensitivity
        from file

        Args:
            size (int): number of samples to generate
            filename (str): file to read values from
            parameters (list): list with parameter names
            seeds (str): default or None
        """
        _raise_if_duplicates(parameters)
        self.sensvalues = pd.DataFrame(columns=parameters, index=range(size))
        extern_values = parameters_from_extern(filename)
        if size > len(extern_values):
            raise ValueError(
                f"Number of realisations {size} specified for "
                f"sensitivity {self.sensname} is larger than rows in "
                f"file {filename}"
            )
        for param in parameters:
            if param in extern_values:
                self.sensvalues[param] = list(extern_values[param][:size])
            else:
                raise ValueError(f"Parameter {param} not in external file")

        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[:size]
