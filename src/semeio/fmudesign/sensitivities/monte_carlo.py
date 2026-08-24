import hashlib
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import probabilit

from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign.config_validation import SeedStrategy
from semeio.fmudesign.quality_report import print_corrmat
from semeio.fmudesign.utils import printwarning

from .sensitivity import Sensitivity

# (group_name, correlation_matrix, member_params)
CorrelationGroup = tuple[str, pd.DataFrame, list[str]]


def _derive_rng(base_seed: int, *keys: str) -> np.random.Generator:
    """Return a numpy Generator seeded from ``base_seed`` and ``keys``.

    Keys are ``(sensname, "param", param_name)`` for an uncorrelated parameter
    and ``(sensname, "corr", group_name)`` for a correlation group.

    Components are length-prefixed so that no two distinct key tuples can hash
    to the same payload, e.g. ``("a", "b:c")`` versus ``("a:b", "c")``.
    """
    hasher = hashlib.sha256()
    for component in (str(base_seed), *keys):
        data = component.encode("utf-8")
        hasher.update(len(data).to_bytes(4, "big"))
        hasher.update(data)
    return np.random.default_rng(int.from_bytes(hasher.digest(), "big"))


class MonteCarloSensitivity(Sensitivity):
    """
    For a MonteCarloSensitivity one or several parameters
    are drawn from specified distributions with or without correlations.
    A MonteCarloSensitivity can only contain
    one case, where the name SENSCASE is automatically set to 'p10_p90' in the
    design matrix to flag that p10_p90 should be calculated in TornadoPlot.

    Attributes:
        sensname (str):  name for the sensitivity.
            Equals SENSNAME in design matrix.
        sensvalues (pd.DataFrame):  parameters and values for the sensitivity
            with realisation numbers as index.
    """

    def generate(
        self,
        *,
        size: int,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
        corrdict: dict[str, Any] | None,
        rng: np.random.Generator,
        correlation_iterations: int = 0,
        seed_strategy: SeedStrategy = SeedStrategy.JOINT,
        base_seed: int | None = None,
    ) -> None:
        """Generates parameter values by drawing from defined distributions.

        Args:
            size (int): number of rows to generate
            parameters (dict): dictionary of parameters and distributions
            values (list): a list of seed values or None
            corrdict (dict): Configuration for correlated parameters. Contains:
                - 'inputfile': Name of Excel file with correlation matrices
                - 'sheetnames': List of sheet names, where each sheet contains a
                correlation matrix. If None, parameters are treated as uncorrelated.
            rng (numpy.random.Generator): Random number generator instance.
              Draws all values under 'joint'. Unused under 'independent'.
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
            seed_strategy (SeedStrategy): How to seed the sampling.
                - 'joint' (default): all parameters are drawn in a single Latin
                  Hypercube Sampling call. Adding, removing or reordering a
                  parameter reshuffles every parameter.
                - 'independent': each uncorrelated parameter and each correlation
                  group is seeded separately from ``base_seed``, so that editing
                  one leaves the others bit-identical. This means independently
                  keyed random streams, not zero empirical correlation: unrelated
                  parameters still show incidental correlation of order
                  1/sqrt(size), exactly as they do under 'joint'.
                  Values are stable only while ``size``, ``base_seed``, the
                  sensitivity name, and the parameter's own name, distribution
                  and correlation group membership are unchanged.
            base_seed (int | None): Root seed that 'independent' derives one
              generator per parameter and per correlation group from. Required
              for 'independent', unused for 'joint'.
        """
        self.sensvalues = pd.DataFrame(
            columns=list(parameters.keys()), index=range(size)
        )
        self.correlation_dfs_: dict[str, pd.DataFrame] = {}  # correlation matrices

        if size < 0:
            raise ValueError(f"Got < 0 samples ({size=})")

        distr_by_name = {}
        for param_name, (dist_name, dist_params, _) in parameters.items():
            # Convert to a probabilit Distribution object
            distr = design_dist.to_probabilit(
                distname=dist_name, dist_parameters=dist_params
            )
            distr_by_name[param_name] = distr

        # Read and validate the correlation groups once, up front. Both seed
        # strategies consume the same groups; only the seeding differs.
        corr_groups = self._load_correlation_groups(parameters, corrdict)

        if seed_strategy == SeedStrategy.JOINT:
            self._sample_joint(
                size=size,
                distr_by_name=distr_by_name,
                corr_groups=corr_groups,
                correlation_iterations=correlation_iterations,
                rng=rng,
            )
        elif seed_strategy == SeedStrategy.INDEPENDENT:
            if base_seed is None:
                raise ValueError(
                    "'base_seed' is required when seed_strategy='independent'"
                )
            self._sample_independent(
                size=size,
                distr_by_name=distr_by_name,
                corr_groups=corr_groups,
                correlation_iterations=correlation_iterations,
                base_seed=base_seed,
            )
        else:
            raise ValueError(
                f"'seed_strategy' must be one of {[s.value for s in SeedStrategy]}, "
                f"got: {seed_strategy!r}"
            )

        for distr_name, distr_obj in distr_by_name.items():
            samples = distr_obj.samples_
            is_numeric = issubclass(samples.dtype.type, np.number)
            if is_numeric and not np.all(np.isfinite(distr_obj.samples_)):
                raise ValueError(
                    f"Sampling produced non-finite values in {distr_name}={distr_obj}\n"
                    "Please review the parameters in the distribution."
                )

            # Discrete distributions are handled in a special way. We map them
            # to Uniform distributions, sample in [0, 1), then map those samples
            # back to the categorical values AFTER sampling. This is so that we
            # can "induce correlations" between categorical values.
            if hasattr(distr_obj, "_values"):
                probabilities = getattr(distr_obj, "_probabilities", None)
                samples = design_dist.quantiles_to_values(
                    quantiles=samples,
                    values=distr_obj._values,
                    probabilities=probabilities,
                )

            self.sensvalues = self.sensvalues.assign(**{distr_name: samples})

        if self.sensname != "background":
            self.sensvalues["SENSNAME"] = self.sensname
            self.sensvalues["SENSCASE"] = "p10_p90"
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues["RMS_SEED"] = seedvalues[:size]

        null_columns = self.sensvalues.isna().any(axis=0)
        if null_columns.any():
            cols_w_null = list(null_columns.loc[lambda ser: ser].index)
            raise ValueError(f"Found NaN values in columns: {cols_w_null}")

    def _load_correlation_groups(
        self,
        parameters: dict[str, Any],
        corrdict: dict[str, Any] | None,
    ) -> list[CorrelationGroup]:
        """Return ``(group_name, df_correlations, member_params)`` per correlation
        group, reading and validating each matrix.

        Single-member groups are skipped (with a warning) and treated as
        uncorrelated. Populates ``self.correlation_dfs_`` as a side effect.
        Shared by both the 'joint' and 'independent' seed strategies.

        Raises:
            ValueError: if a parameter is a member of more than one correlation
                group, since only one of the requested correlations could then
                be honoured.
        """
        if not corrdict:
            return []

        df_params = (
            pd.DataFrame.from_dict(
                parameters,
                orient="index",
                columns=["dist_name", "dist_params", "corr_sheet"],
            )
            .reset_index()
            .rename(columns={"index": "param_name"})
            .assign(corr_sheet=lambda df: df.corr_sheet.fillna("nocorr"))
        )

        groups = dict(iter(df_params.groupby("corr_sheet")))
        groups.pop("nocorr", None)

        loaded: list[CorrelationGroup] = []
        group_of_param: dict[str, str] = {}
        for corr_group_name, corr_group in groups.items():
            corr_group_name = cast("str", corr_group_name)

            # A single correlation - print warning and skip it
            if len(corr_group) == 1:
                printwarning(corr_group_name)
                continue

            # The Excel sheet only fills in the lower triangle, which
            # read_correlations mirrors into a full symmetric matrix.
            df_correlations = design_dist.read_correlations(
                excel_filename=corrdict["inputfile"], corr_sheet=corr_group_name
            )
            multivariate_parameters = df_correlations.index.tolist()
            correlations = df_correlations.to_numpy()

            # Each group is sampled as one unit, so a parameter in two groups
            # would get the correlations of whichever group is sampled last.
            for name in multivariate_parameters:
                if name in group_of_param:
                    raise ValueError(
                        f"Parameter {name!r} is part of several correlation "
                        f"groups: {group_of_param[name]!r} and "
                        f"{corr_group_name!r}. A parameter may only appear in "
                        "one correlation matrix."
                    )
                group_of_param[name] = corr_group_name

            if self.verbosity == 0:
                print(
                    f"Sampling {len(multivariate_parameters)} parameters",
                    f"in correlation group {corr_group_name!r}",
                )
            else:
                print(
                    f"Sampling {len(multivariate_parameters)} parameters",
                    f"in correlation group {corr_group_name!r}: "
                    f"{multivariate_parameters}",
                )

            # Get the nearest correlation matrix
            nearest = probabilit.correlation.nearest_correlation_matrix(
                correlations, weights=None, eps=1e-6, verbose=False
            )
            if not np.allclose(correlations, nearest):
                print(
                    f"\nWarning: Correlation matrix {corr_group_name!r} is inconsistent"
                )
                print("Requirements:")
                print("  - All diagonal elements must be 1")
                print("  - All elements must be between -1 and 1")
                print("  - The matrix must be positive semi-definite")
                print("\nInput correlation matrix:")
                print_corrmat(df_correlations)
                df_correlations = pd.DataFrame(
                    nearest,
                    index=df_correlations.index,
                    columns=df_correlations.columns,
                )
                print("\nAdjusted to nearest consistent correlation matrix:")
                print_corrmat(df_correlations)

            self.correlation_dfs_[corr_group_name] = df_correlations
            loaded.append((corr_group_name, df_correlations, multivariate_parameters))

        return loaded

    @staticmethod
    def _make_correlator(
        correlation_iterations: int, rng: np.random.Generator
    ) -> probabilit.correlation.Correlator:
        """Return the correlator used to induce the requested correlations.

        ``correlation_iterations=0`` gives plain Iman-Conover. A positive number
        adds that many rounds of random row swaps on top, keeping only the swaps
        that move the observed correlation closer to the target. The result is
        therefore never further off than Iman-Conover alone, and usually closer.
        """
        if correlation_iterations > 0:
            return probabilit.correlation.Composite(
                iterations=correlation_iterations,
                correlation_type="pearson",
                random_state=rng,
                verbose=False,
            )
        return probabilit.correlation.ImanConover()

    def _sample_joint(
        self,
        *,
        size: int,
        distr_by_name: dict[str, Any],
        corr_groups: list[CorrelationGroup],
        correlation_iterations: int,
        rng: np.random.Generator,
    ) -> None:
        """Draw all parameters in a single LHS call sharing one RNG, so the
        sample of every parameter depends on the full parameter set.
        """
        # Create a dummy NoOp node for sampling each parent distribution
        expression = probabilit.modeling.NoOp(*distr_by_name.values())

        for _name, df_correlations, member_params in corr_groups:
            corrvars = [distr_by_name[name] for name in member_params]
            expression.correlate(*corrvars, corr_mat=df_correlations.to_numpy())

        correlator = self._make_correlator(correlation_iterations, rng)

        # Sample the dummy node. This samples every parent distribution and
        # stores the draws on each distribution object as 'samples_'.
        expression.sample(
            size=size, random_state=rng, method="lhs", correlator=correlator
        )

    def _sample_independent(
        self,
        *,
        size: int,
        distr_by_name: dict[str, Any],
        corr_groups: list[CorrelationGroup],
        correlation_iterations: int,
        base_seed: int,
    ) -> None:
        """Sample each correlation group and each uncorrelated parameter as a
        separate unit, seeded from ``base_seed`` and a stable key. Adding,
        removing or reordering a parameter therefore leaves the other
        parameters unchanged (given a fixed seed and sample size).
        """
        grouped_params: set[str] = set()

        # Each correlation group is one independently seeded unit.
        for group_name, df_correlations, member_params in corr_groups:
            distrs = [distr_by_name[name] for name in member_params]
            expression = probabilit.modeling.NoOp(*distrs)
            expression.correlate(*distrs, corr_mat=df_correlations.to_numpy())
            unit_rng = _derive_rng(base_seed, self.sensname, "corr", group_name)
            correlator = self._make_correlator(correlation_iterations, unit_rng)
            expression.sample(
                size=size, random_state=unit_rng, method="lhs", correlator=correlator
            )
            grouped_params.update(member_params)

        # Each remaining (uncorrelated) parameter is its own independent unit.
        for param_name, distr in distr_by_name.items():
            if param_name in grouped_params:
                continue
            unit_rng = _derive_rng(base_seed, self.sensname, "param", param_name)
            distr.sample(size=size, random_state=unit_rng, method="lhs")
