"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.


A DesignMatrix is a "God-object" that contains information about all info
used to generate design matrices, including one or several Sensitivities.


"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

import semeio
from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign.config_validation import SeedStrategy, validate_configuration
from semeio.fmudesign.quality_report import QualityReporter
from semeio.fmudesign.sensitivities import (
    BackgroundSensitivity,
    ExternSensitivity,
    MonteCarloSensitivity,
    ScenarioSensitivity,
    ScenarioSensitivityCase,
    SeedSensitivity,
    Sensitivity,
    SingleRealisationReference,
)
from semeio.fmudesign.utils import (
    find_max_realisations,
    parameters_from_extern,
    to_numeric_safe,
)

if TYPE_CHECKING:
    from collections.abc import Hashable


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a onebyone design
    or a full montecarlo design.

    Attributes:
        designvalues (pd.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (dict): default values for design
        backgroundvalues (pd.DataFrame): Used when background parameters are
            not constant. Either a set is sampled from specified distributions
            or they are read from a file.
    """

    def __init__(self, verbosity: int = 0, output_dir: Path | None = None) -> None:
        """
        Placeholders for:
        designvalues: dataframe with parameters that varies
        defaultvalues: dictionary of default/base case values
        backgroundvalues: dataframe with background parameters
        seedvalues: list of seed values
        verbosity: how much information to print
        output_dir: where to write debugging output and QC plots
        rng: generator seeded with 'distribution_seed'; draws all parameters
            under 'joint'
        seed_strategy: 'joint' or 'independent' Monte Carlo seeding
        base_seed: root seed that 'independent' derives one generator per
            parameter and per correlation group from. Equals
            'distribution_seed', or a draw from rng when no seed is given.
            Unused under 'joint'

        """
        self.designvalues: pd.DataFrame
        self.defaultvalues: dict[Hashable, Any] = {}
        self.backgroundvalues: pd.DataFrame | None = None
        self.seedvalues: list[int] | None = None
        self.verbosity: int = verbosity
        self.output_dir: Path | None = output_dir
        self.rng: np.random.Generator
        self.seed_strategy: SeedStrategy
        self.base_seed: int

    def reset(self) -> None:
        """Resets DesignMatrix to empty. Necessary iin case method generate
        is used several times for same instance of DesignMatrix"""
        self.designvalues = pd.DataFrame()
        self.defaultvalues = {}
        self.backgroundvalues = None
        self.seedvalues = None

    def generate(self, inputdict: dict[str, Any]) -> None:
        """Generating design matrix from input dictionary in specific
        format. Adding default values and background values if existing.
        Looping through sensitivities and adding them to designvalues.

        Args:
            inputdict (dict): input parameters for design
        """
        inputdict = validate_configuration(inputdict, verbosity=self.verbosity)

        self.reset()  # Emptying if regenerating matrix
        self.rng = np.random.default_rng(seed=inputdict.get("distribution_seed"))
        self.defaultvalues = inputdict["defaultvalues"]

        self.seed_strategy = inputdict["seed_strategy"]
        distribution_seed = inputdict.get("distribution_seed")
        self.base_seed = (
            distribution_seed
            if distribution_seed is not None
            else int(self.rng.integers(2**63))
        )

        # Reading or generating rms seed values
        max_reals = find_max_realisations(inputdict)
        self.seedvalues = DesignMatrix.create_rms_seeds(inputdict["seeds"], max_reals)

        # If background values used - read or generate
        if "background" in inputdict:
            self.add_background(
                back_dict=inputdict["background"],
                max_values=max_reals,
                correlation_iterations=inputdict.get("correlation_iterations", 0),
            )

        sensitivity: Sensitivity

        self.designvalues["SENSNAME"] = None
        self.designvalues["SENSCASE"] = None

        for key in inputdict["sensitivities"]:
            sens = inputdict["sensitivities"][key]

            # Numer of realization (rows) to use for each sensitivity
            size = sens["numreal"] if "numreal" in sens else inputdict["repeats"]

            print(f" Generating sensitivity : {key}")

            if sens["senstype"] == "ref":
                sensitivity = SingleRealisationReference(key, verbosity=self.verbosity)
                sensitivity.generate(size=size)
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "background":
                sensitivity = BackgroundSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(size=size)
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "seed":
                sensitivity = SeedSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    seedname=sens["seedname"],
                    seedvalues=self.seedvalues,
                    parameters=sens["parameters"],
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "scenario":
                sensitivity = ScenarioSensitivity(key, verbosity=self.verbosity)
                for casekey in sens["cases"]:
                    case = sens["cases"][casekey]
                    temp_case = ScenarioSensitivityCase(casekey)
                    temp_case.generate(
                        size=size,
                        parameters=case,
                        seedvalues=self.seedvalues,
                    )
                    sensitivity.add_case(temp_case)
                    sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "dist":
                sensitivity = MonteCarloSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                    corrdict=sens["correlations"],
                    rng=self.rng,
                    correlation_iterations=inputdict.get("correlation_iterations", 0),
                    seed_strategy=self.seed_strategy,
                    base_seed=self.base_seed,
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)

            elif sens["senstype"] == "extern":
                sensitivity = ExternSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    filename=sens["extern_file"],
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)

            else:
                raise ValueError(f"Unknown sensitivity type: {sens['senstype']!r}")

            # MonteCarloSensitivity is special - it can produce debugging outputs
            is_montecarlo = isinstance(sensitivity, MonteCarloSensitivity)
            if is_montecarlo and self.verbosity > 0:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                quality_reporter = QualityReporter(
                    df=sensitivity.sensvalues, variables=sens["parameters"]
                )

                # Print to terminal
                quality_reporter.print_numeric()
                quality_reporter.print_discrete()
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    quality_reporter.print_correlation(corr_name, df_corr)

            if is_montecarlo and self.verbosity > 1 and self.output_dir is not None:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                output_dir = self.output_dir / key
                quality_reporter.plot_columns(output_dir=output_dir)

                # Correlations
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    # Always plot heatmaps
                    quality_reporter.plot_correlation_heatmap(
                        corr_name, df_corr, output_dir=output_dir, show=False
                    )

                    # Only plot pairgrid for small correlations
                    if len(df_corr) <= 6:
                        quality_reporter.plot_correlation(
                            corr_name, df_corr, output_dir=output_dir, show=False
                        )

        # Once all sensitivities have been added, complete the work
        if "background" in inputdict:
            self._fill_with_background_values()
        self._fill_with_defaultvalues()

        # Round columns in `self.designvalues` to desired precision
        self._set_decimals(inputdict)

        # Create REAL column (realization number)
        self.designvalues = self.designvalues.assign(REAL=lambda df: np.arange(len(df)))

        # Re-order columns
        start_cols = ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]
        self.designvalues = self.designvalues[
            [col for col in start_cols if col in self.designvalues]
            + [col for col in self.designvalues if col not in start_cols]
        ]

        # Make all values numerical if possible
        self.designvalues = self.designvalues.map(to_numeric_safe)

    def to_xlsx(
        self,
        filename: str,
        designsheet: str = "DesignSheet01",
        defaultsheet: str = "DefaultValues",
    ) -> None:
        """Writing design matrix to excel workfbook on standard fmu format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
        """
        # Create folder for output file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
            print(f"Warning: Missing .xlsx suffix. Changed to: {filename}")

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            self.designvalues.to_excel(
                writer, sheet_name=designsheet, index=False, header=True
            )
            # Default values
            defaults = pd.DataFrame(
                data=list(self.defaultvalues.items()),
                columns=["defaultparameters", "defaultvalue"],
            )
            defaults.to_excel(
                writer, sheet_name=defaultsheet, index=False, header=False
            )

            version_info = pd.DataFrame(
                {
                    "Description": ["Created using semeio version:", "Created on:"],
                    "Value": [
                        semeio.__version__,
                        datetime.now()
                        .astimezone()
                        .isoformat(sep=" ", timespec="seconds"),
                    ],
                }
            )
            version_info.to_excel(writer, sheet_name="Metadata", index=False)

        print(
            f"Design matrix of shape {self.designvalues.shape} written to: {filename!r}"
        )

    @staticmethod
    def create_rms_seeds(seeds: list | str | None, max_reals: int) -> list | None:
        """Create RMS seems from 'seeds' argument.

        Args:
            seeds: Seed configuration. Can be:
                - None: returns None
                - "default": Generates sequential seeds 1001, 1002, 1003, ...
                - list of seeds, e.g. [1, 2, 3]
            max_reals: Maximum number of seed values to generate or load

        Examples
        --------
        >>> DesignMatrix.create_rms_seeds([1, 2, 3], max_reals=5)
        Provided number of seed values (3) in external file is lower than the maximum number of realisations (5).
         Seeds will be repeated, e.g. [1, 2, 3] => [1, 2, 3, 1, 2, ...]
        [1, 2, 3, 1, 2]
        """  # ruff: ignore[line-too-long]
        if seeds is None:
            return None

        if seeds == "default":
            return [item + 1000 for item in range(max_reals)]

        if isinstance(seeds, list):
            if max_reals > len(seeds):
                print(
                    f"Provided number of seed values ({len(seeds)}) in external file "
                    f"is lower than the maximum number of realisations ({max_reals}).\n"
                    " Seeds will be repeated, e.g. [1, 2, 3] => [1, 2, 3, 1, 2, ...]"
                )

            return [seeds[item % len(seeds)] for item in range(max_reals)]

        # Raise if none of the cases above apply. We do this because if we did not we
        # would return None, which is a valid case in itself.
        raise ValueError(f"Must be None, 'default' or list: {seeds=}")

    def add_background(
        self,
        back_dict: dict[str, Any] | None,
        max_values: int,
        correlation_iterations: int = 0,
    ) -> None:
        """Adding background as specified in dictionary.
        Either from external file or from distributions in background
        dictionary

        Seeding follows ``self.seed_strategy`` / ``self.base_seed``, which are
        set by :meth:`generate`.

        Args:
            back_dict (dict): how to generate background values
            max_values (int): number of background values to generate
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        if back_dict is None:
            self.backgroundvalues = None
        elif "extern" in back_dict:
            print(f"Reading background values from: {back_dict['extern']}")
            self.backgroundvalues = parameters_from_extern(back_dict["extern"])
        elif "parameters" in back_dict:
            print("Generating background values from distributions.")
            self._add_dist_background(
                back_dict=back_dict,
                size=max_values,
                correlation_iterations=correlation_iterations,
            )

    def background_to_excel(
        self, filename: str, backgroundsheet: str = "Background"
    ) -> None:
        """Writing background values to an Excel spreadsheet

        Args:
            filename (str): output filename (extension .xlsx)
            backgroundsheet (str): name of excel sheet
        """
        if self.backgroundvalues is None:
            raise ValueError("No background values available to write to Excel")

        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.backgroundvalues.to_excel(
            xlsxwriter, sheet_name=backgroundsheet, index=False, header=True
        )
        xlsxwriter.close()
        print(f"Backgroundvalues written to {filename}")

    def _add_sensitivity(
        self,
        sensitivity: Sensitivity,
    ) -> None:
        """Adding a sensitivity to the design

        Args:
            sensitivity of class Scenario, MonteCarlo or Extern
        """
        existing_values = self.designvalues
        new_values = sensitivity.sensvalues
        self.designvalues = pd.concat([existing_values, new_values])

    def _fill_with_background_values(self) -> None:
        """Substituting NaNs with background values if existing.
        background values not in design are added as separate columns
        """
        if self.backgroundvalues is None:
            return

        grouped = self.designvalues.groupby(["SENSNAME", "SENSCASE"], sort=False)
        result_values = pd.DataFrame()
        for sensname, case_ in grouped:
            temp_df = case_.reset_index()
            temp_df = temp_df.fillna(self.backgroundvalues)
            for key in self.backgroundvalues.columns:
                if key not in case_:
                    temp_df[key] = self.backgroundvalues[key]
                    if len(temp_df) > len(self.backgroundvalues):
                        raise ValueError(
                            "Provided number of background values "
                            f"{len(self.backgroundvalues)} is smaller than number"
                            f" of realisations for sensitivity {sensname}"
                        )
                elif len(temp_df) > len(self.backgroundvalues):
                    print(
                        "Provided number of background values "
                        f"({len(self.backgroundvalues)}) is smaller than number"
                        f" of realisations for sensitivity {sensname}"
                        f" and parameter {key}. "
                        "Will be filled with default values."
                    )
            existing_values = result_values.copy()
            result_values = pd.concat([existing_values, temp_df])

        result_values = result_values.drop(["index"], axis=1)
        self.designvalues = result_values

    def _fill_with_defaultvalues(self) -> None:
        """Filling NaNs with default values"""
        for key in self.designvalues.columns:
            if key in self.defaultvalues:
                self.designvalues[key] = self.designvalues[key].fillna(
                    self.defaultvalues[key]
                )
            elif key not in {"REAL", "SENSNAME", "SENSCASE", "RMS_SEED"}:
                raise LookupError(f"No defaultvalues given for parameter {key} ")

    def _add_dist_background(
        self,
        back_dict: dict[str, Any],
        size: int,
        correlation_iterations: int,
    ) -> None:
        """Drawing background values from distributions
        specified in dictionary

        Args:
            back_dict (dict): parameters and distributions
            size (int): Number of samples to generate
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """

        mc_background = MonteCarloSensitivity("background")
        mc_background.generate(
            size=size,
            parameters=back_dict["parameters"],
            seedvalues=None,
            corrdict=back_dict["correlations"],
            rng=self.rng,
            correlation_iterations=correlation_iterations,
            seed_strategy=self.seed_strategy,
            base_seed=self.base_seed,
        )
        mc_backgroundvalues = mc_background.sensvalues.copy()
        quality_reporter = QualityReporter(
            df=mc_backgroundvalues, variables=back_dict["parameters"]
        )

        # Print info to terminal
        if self.verbosity > 0:
            quality_reporter.print_numeric()
            quality_reporter.print_discrete()
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.print_correlation(corr_name, df_corr)

        # Write plots to disk
        if self.verbosity > 0 and self.output_dir is not None:
            output_dir = self.output_dir / mc_background.sensname
            quality_reporter.plot_columns(output_dir=output_dir)

            # Correlations
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.plot_correlation(
                    corr_name, df_corr, output_dir=output_dir, show=False
                )

        # Rounding of background values as specified
        if "decimals" in back_dict:
            for key in back_dict["decimals"]:
                if design_dist.is_number(mc_backgroundvalues[key].iloc[0]):
                    mc_backgroundvalues[key] = (
                        mc_backgroundvalues[key]
                        .astype(float)
                        .round(int(back_dict["decimals"][key]))
                    )
                else:
                    raise ValueError("Cannot round a string parameter")
        self.backgroundvalues = mc_backgroundvalues.copy()

    def _set_decimals(self, inputdict: dict[str, Any]) -> None:
        """Round to specified number of decimals.

        Args:
            inputdict (dictionary): input diction that might have a sub-dict
                                    with key "decimals". This sub-dict has
                                    (key, value)s are (param, decimals)
        """
        inputdict = copy.deepcopy(inputdict)

        # No decimal information => Nothing to do.
        if not inputdict.get("decimals", {}):
            return

        # If there are dependencies (derived params) that are copies,
        # like TO := copy(FROM), then the new TO column must be rounded too.
        for sensdict in inputdict["sensitivities"].values():
            if not sensdict["dependencies"]:
                continue
            for from_param, from_dict in sensdict["dependencies"].items():
                for to_param in from_dict["to_params"]:
                    if not inputdict["decimals"].get(from_param, None):
                        continue
                    inputdict["decimals"][to_param] = inputdict["decimals"].get(
                        from_param, ""
                    )

        # Round each column
        dict_decimals = inputdict["decimals"]
        for key in self.designvalues.columns:
            if key in dict_decimals:
                if design_dist.is_number(self.designvalues[key].iloc[0]):
                    self.designvalues[key] = (
                        self.designvalues[key]
                        .astype(float)
                        .round(int(dict_decimals[key]))
                    )
                else:
                    raise ValueError(f"Cannot round a string parameter {key}")
