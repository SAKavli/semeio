import pandas as pd

from .sensitivity import Sensitivity


class BackgroundSensitivity(Sensitivity):
    """
    The class is used in set-ups where one sensitivities
    are run on top of varying background parameters.
    Typically used when RMS_SEED is not a parameter, so the reference
    for tornadoplots will be the realisations with all parameters
    at their default values except the background parameters.
    SENSCASE will be set to 'p10_p90' in design matrix.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(self, size: int) -> None:
        """Generates realisation number only

        Args:
            size (int): number of rows to generate
        """
        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"
