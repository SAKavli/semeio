import pandas as pd

from .sensitivity import Sensitivity


class SingleRealisationReference(Sensitivity):
    """
    The class is used in set-ups where one wants a single realisation
    containing only default values as a reference, but the realisation
    itself is not included in a sensitivity.
    Typically used when RMS_SEED is not a parameter.
    SENSCASE will be set to 'ref' in design matrix, to flag that it should be
    excluded as a sensitivity in the plot.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
    ) -> None:
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "ref"
