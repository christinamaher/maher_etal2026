"""
Core neural analysis functions.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.api import OLS
import statsmodels.formula.api as smf
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Permutation regression
# -----------------------------------------------------------------------------

@dataclass
class PermutationRegressionResult:
    """Container for permutation-regression outputs.

    Attributes
    ----------
    summary : pandas.DataFrame
        Table with observed coefficients, permutation-null means/SDs,
        z-scores, p-values, and model log likelihood.
    permuted_params : numpy.ndarray
        Coefficients from each permuted model with shape
        ``(n_permutations, n_coefficients)``.
    original_model : statsmodels RegressionResultsWrapper
        Fitted OLS model from the original, non-permuted data.
    """

    summary: pd.DataFrame
    permuted_params: np.ndarray
    original_model: object


def fit_ols_params(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit an OLS model and return regression coefficients.

    Parameters
    ----------
    y : ndarray, shape (n_observations,)
        Dependent variable.
    x : ndarray, shape (n_observations, n_predictors)
        Design matrix.

    Returns
    -------
    ndarray
        OLS parameter estimates.
    """

    return OLS(y, x).fit().params


def permutation_regression_zscore(
    data: pd.DataFrame,
    formula: str,
    regressor_to_permute: str,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> PermutationRegressionResult:
    """Run OLS and compute permutation-derived coefficient z-scores.

    The original model is fit using ``formula``. A null distribution is built by
    shuffling one design-matrix column while keeping the remaining design matrix
    intact.

    Parameters
    ----------
    data : pandas.DataFrame
        Observation-level dataframe containing all variables in ``formula``.
    formula : str
        Patsy formula for the OLS model.
    regressor_to_permute : str
        Design-matrix column to shuffle and test. For categorical predictors,
        this should match the exact Patsy-expanded column name.
    n_permutations : int, default=1000
        Number of permutations used to build the null distribution.
    random_state : int or None, default=None
        Optional random seed.
    show_progress : bool, default=True
        Whether to show a tqdm progress bar.

    Returns
    -------
    PermutationRegressionResult
        Summary table, permuted coefficients, and original fitted model.
    """

    rng = np.random.default_rng(random_state)
    y, x = patsy.dmatrices(formula, data, return_type="dataframe")

    if regressor_to_permute not in x.columns:
        raise ValueError(
            f"regressor_to_permute='{regressor_to_permute}' was not found. "
            f"Available design-matrix columns are: {list(x.columns)}"
        )

    original_model = OLS(y, x).fit()
    original_params = original_model.params
    y_values = y.values.ravel()

    iterator = range(n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutations")

    permuted_params = []
    for _ in iterator:
        x_perm = x.copy()
        x_perm[regressor_to_permute] = rng.permutation(
            x_perm[regressor_to_permute].values
        )
        permuted_params.append(fit_ols_params(y_values, x_perm.values))

    permuted_params = np.asarray(permuted_params)
    permuted_means = np.mean(permuted_params, axis=0)
    permuted_stds = np.std(permuted_params, axis=0, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = (original_params - permuted_means) / permuted_stds

    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    summary = pd.DataFrame(
        {
            "original_estimate": original_params,
            "permuted_mean": permuted_means,
            "permuted_std": permuted_stds,
            "z_score": z_scores,
            "p_value": p_values,
            "log_likelihood": original_model.llf,
        },
        index=original_params.index,
    )

    return PermutationRegressionResult(summary, permuted_params, original_model)


def run_permutation_regression_by_group(
    data: pd.DataFrame,
    group_col: str,
    formula: str,
    regressor_to_permute: str,
    subject_col: Optional[str] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Run permutation regression separately for each group.

    Use this for per-electrode, per-channel, or per-unit analyses.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    group_col : str
        Column defining independent groups, such as electrode or unit ID.
    formula : str
        Patsy formula for the OLS model.
    regressor_to_permute : str
        Design-matrix column to shuffle and extract.
    subject_col : str or None, default=None
        Optional subject ID column to carry into the output.
    n_permutations : int, default=1000
        Number of permutations per group.
    random_state : int or None, default=None
        Optional random seed.

    Returns
    -------
    pandas.DataFrame
        One row per group with original coefficient, z-score, p-value, and
        optional subject identifier.
    """

    rows = []
    for idx, group_value in enumerate(data[group_col].dropna().unique()):
        group_df = data.loc[data[group_col] == group_value].copy()
        seed = None if random_state is None else random_state + idx

        try:
            result = permutation_regression_zscore(
                group_df,
                formula=formula,
                regressor_to_permute=regressor_to_permute,
                n_permutations=n_permutations,
                random_state=seed,
                show_progress=False,
            )
        except ValueError:
            continue

        summary = result.summary
        if regressor_to_permute not in summary.index:
            continue

        row = {
            "group": group_value,
            "original_estimate": summary.loc[regressor_to_permute, "original_estimate"],
            "z_score": summary.loc[regressor_to_permute, "z_score"],
            "p_value": summary.loc[regressor_to_permute, "p_value"],
            "log_likelihood": summary.loc[regressor_to_permute, "log_likelihood"],
        }

        if subject_col is not None and subject_col in group_df.columns:
            row["subject"] = group_df[subject_col].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Time-resolved regression
# -----------------------------------------------------------------------------

def time_resolved_regression_single_channel(
    data: pd.DataFrame,
    formula: str,
    time_col: str,
    regressor_to_permute: Optional[str] = None,
    permute: bool = False,
    n_permutations: int = 100,
    random_state: Optional[int] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Fit the same OLS model independently at each time point.

    The input must already be in long format, with one row per observation per
    time point. This function does not reshape or format data.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dataframe containing ``time_col`` and all variables in
        ``formula``.
    formula : str
        Patsy formula for the OLS model.
    time_col : str
        Column containing timepoint/window identifiers.
    regressor_to_permute : str or None, default=None
        Design-matrix column to shuffle if ``permute=True``.
    permute : bool, default=False
        If True, compute permutation-derived z-scores at each time point.
    n_permutations : int, default=100
        Number of permutations per timepoint when ``permute=True``.
    random_state : int or None, default=None
        Optional seed.
    show_progress : bool, default=False
        Whether to show a tqdm progress bar.

    Returns
    -------
    pandas.DataFrame
        Long-format regression results with one row per coefficient per
        timepoint.
    """

    if permute and regressor_to_permute is None:
        raise ValueError("regressor_to_permute is required when permute=True.")

    all_results = []
    times = np.sort(data[time_col].dropna().unique())
    iterator = tqdm(times, desc="Time-resolved regression") if show_progress else times

    for idx, time_value in enumerate(iterator):
        one_time_df = data.loc[data[time_col] == time_value].copy()

        if permute:
            seed = None if random_state is None else random_state + idx
            perm_result = permutation_regression_zscore(
                one_time_df,
                formula=formula,
                regressor_to_permute=regressor_to_permute,
                n_permutations=n_permutations,
                random_state=seed,
                show_progress=False,
            )
            result_df = perm_result.summary.reset_index().rename(
                columns={
                    "index": "parameter",
                    "original_estimate": "beta",
                    "p_value": "permutation_p",
                    "z_score": "permutation_z",
                }
            )
        else:
            y, x = patsy.dmatrices(formula, one_time_df, return_type="dataframe")
            fit = OLS(y, x).fit()
            result_df = pd.DataFrame(
                {
                    "parameter": fit.params.index,
                    "beta": fit.params.values,
                    "standard_error": fit.bse.values,
                    "p_value": fit.pvalues.values,
                }
            )

        result_df[time_col] = time_value
        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True)


def extract_parameter_trajectory(
    regression_results: pd.DataFrame,
    parameter: str,
    time_col: str,
    beta_col: str = "beta",
) -> pd.DataFrame:
    """Extract a single coefficient trajectory from time-resolved results."""

    return (
        regression_results.loc[
            regression_results["parameter"] == parameter,
            [time_col, beta_col],
        ]
        .rename(columns={beta_col: "beta"})
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Cluster permutation tests
# -----------------------------------------------------------------------------

def run_one_sample_cluster_test(
    data: np.ndarray,
    n_permutations: int = 1024,
    tail: int = 0,
    seed: Optional[int] = None,
    **mne_kwargs,
) -> Dict[str, object]:
    """Run MNE one-sample cluster permutation test against zero.

    Parameters
    ----------
    data : ndarray
        Subject-level data. For TFR analyses, expected shape is typically
        ``(subjects, frequencies, times)``. For time courses, expected shape is
        typically ``(subjects, times)``.
    n_permutations : int, default=1024
        Number of permutations.
    tail : {-1, 0, 1}, default=0
        Test direction. ``0`` is two-sided.
    seed : int or None, default=None
        Random seed.
    **mne_kwargs
        Additional options passed to
        ``mne.stats.permutation_cluster_1samp_test``.

    Returns
    -------
    dict
        Contains ``t_obs``, ``clusters``, ``cluster_p_values``, and ``h0``.
    """

    from mne.stats import permutation_cluster_1samp_test

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
        np.asarray(data),
        n_permutations=n_permutations,
        tail=tail,
        seed=seed,
        **mne_kwargs,
    )
    return {
        "t_obs": t_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "h0": h0,
    }


def run_two_sample_cluster_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 1024,
    tail: int = 0,
    seed: Optional[int] = None,
    **mne_kwargs,
) -> Dict[str, object]:
    """Run MNE two-sample cluster permutation test.

    Parameters
    ----------
    group_a, group_b : ndarray
        Arrays for the two groups. For time courses, expected shape is usually
        ``(observations, times)``.
    n_permutations : int, default=1024
        Number of permutations.
    tail : {-1, 0, 1}, default=0
        Test direction. ``0`` is two-sided.
    seed : int or None, default=None
        Random seed.
    **mne_kwargs
        Additional options passed to ``mne.stats.permutation_cluster_test``.

    Returns
    -------
    dict
        Contains ``t_obs``, ``clusters``, ``cluster_p_values``, and ``h0``.
    """

    from mne.stats import permutation_cluster_test

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
        [np.asarray(group_a), np.asarray(group_b)],
        n_permutations=n_permutations,
        tail=tail,
        seed=seed,
        **mne_kwargs,
    )
    return {
        "t_obs": t_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "h0": h0,
    }


def summarize_clusters(
    cluster_results: Dict[str, object],
    axis_values: Sequence[np.ndarray],
    axis_names: Sequence[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize significant clusters returned by MNE.

    Parameters
    ----------
    cluster_results : dict
        Output from one of the cluster-test functions above.
    axis_values : sequence of arrays
        Coordinate values for each non-observation axis, such as ``[freqs, times]``
        or ``[times]``.
    axis_names : sequence of str
        Names corresponding to ``axis_values``, such as ``["freq", "time"]``.
    alpha : float, default=0.05
        Significance threshold for cluster p-values.

    Returns
    -------
    pandas.DataFrame
        One row per significant cluster.
    """

    rows = []

    for cluster_idx, (cluster, p_value) in enumerate(
        zip(cluster_results["clusters"], cluster_results["cluster_p_values"])
    ):
        if p_value >= alpha:
            continue

        if isinstance(cluster, tuple):
            cluster_indices = cluster
        else:
            cluster_indices = np.where(cluster)

        row = {
            "cluster": cluster_idx,
            "p_value": float(p_value),
            "n_points": int(len(cluster_indices[0])),
        }

        for dim_name, dim_values, dim_indices in zip(axis_names, axis_values, cluster_indices):
            row[f"{dim_name}_min"] = float(np.asarray(dim_values)[np.min(dim_indices)])
            row[f"{dim_name}_max"] = float(np.asarray(dim_values)[np.max(dim_indices)])

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Mixed-effects models
# -----------------------------------------------------------------------------

def fit_intercept_only_mixed_model(
    data: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    reml: bool = True,
):
    """Fit an intercept-only mixed model testing whether outcome differs from 0."""

    model = smf.mixedlm(f"{outcome_col} ~ 1", data=data, groups=data[group_col])
    return model.fit(reml=reml)


def fit_mixed_model(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
    reml: bool = True,
    missing: str = "drop",
):
    """Fit a mixed-effects model with a user-specified formula."""

    model = smf.mixedlm(formula=formula, data=data, groups=data[group_col], missing=missing)
    return model.fit(reml=reml)


# -----------------------------------------------------------------------------
# General summary/statistical utilities
# -----------------------------------------------------------------------------

def zscore_within_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Z-score a column within each group."""

    out = df.copy()
    if output_col is None:
        output_col = value_col

    out[output_col] = out.groupby(group_col)[value_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    return out


def extract_peak_values(
    values: np.ndarray,
    axis_coordinates: np.ndarray,
    mode: str = "negative",
    window: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Extract peak magnitude and location for each row of a 2D array.

    Parameters
    ----------
    values : ndarray, shape (observations, axis)
        Values to search.
    axis_coordinates : ndarray, shape (axis,)
        Coordinates corresponding to the second dimension of ``values``.
    mode : {"negative", "positive"}
        Whether to extract the most negative or most positive peak.
    window : tuple or None, default=None
        Optional coordinate window.

    Returns
    -------
    pandas.DataFrame
        Columns ``peak_value`` and ``peak_location``.
    """

    values = np.asarray(values)
    axis_coordinates = np.asarray(axis_coordinates)

    if window is not None:
        mask = (axis_coordinates >= window[0]) & (axis_coordinates <= window[1])
        values = values[:, mask]
        local_coordinates = axis_coordinates[mask]
    else:
        local_coordinates = axis_coordinates

    if mode == "negative":
        peak_idx = np.argmin(values, axis=1)
    elif mode == "positive":
        peak_idx = np.argmax(values, axis=1)
    else:
        raise ValueError("mode must be 'negative' or 'positive'.")

    return pd.DataFrame(
        {
            "peak_value": values[np.arange(values.shape[0]), peak_idx],
            "peak_location": local_coordinates[peak_idx],
        }
    )


def ks_test(group_a: Sequence[float], group_b: Sequence[float]):
    """Run a two-sample Kolmogorov-Smirnov test."""

    return stats.ks_2samp(group_a, group_b)


def compute_difference(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """Compute elementwise difference between two arrays."""

    return np.asarray(array_a) - np.asarray(array_b)


def summarize_significant_units(
    results: pd.DataFrame,
    p_col: str,
    beta_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize counts/proportions of significant positive/negative effects."""

    n_total = len(results)
    if n_total == 0:
        return pd.DataFrame(
            [{
                "n_total": 0,
                "n_significant": 0,
                "n_positive": 0,
                "n_negative": 0,
                "prop_significant": np.nan,
                "prop_positive": np.nan,
                "prop_negative": np.nan,
            }]
        )

    significant = results[p_col] < alpha
    positive = significant & (results[beta_col] > 0)
    negative = significant & (results[beta_col] < 0)

    return pd.DataFrame(
        [{
            "n_total": n_total,
            "n_significant": int(significant.sum()),
            "n_positive": int(positive.sum()),
            "n_negative": int(negative.sum()),
            "prop_significant": float(significant.mean()),
            "prop_positive": float(positive.mean()),
            "prop_negative": float(negative.mean()),
        }]
    )


def binomial_test_count(
    n_successes: int,
    n_observations: int,
    null_proportion: float,
    alternative: str = "greater",
):
    """Run an exact binomial test for a count against a null proportion."""

    return stats.binomtest(
        n_successes,
        n_observations,
        p=null_proportion,
        alternative=alternative,
    )
