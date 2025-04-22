import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import itertools
import statsmodels.api as sm
from scipy.stats import sem, t
from typing import List, Dict


import re


def get_missing_values(df=pd.DataFrame):
    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_values_pc = 100 * missing_values / df.shape[0]
    res = pd.concat(
        [missing_values, missing_values_pc], keys=["Count", "Percent"], axis=1
    )
    return res


def conditional_cramersV(var1: pd.Series, var2: pd.Series, alpha: float = 0.05):
    """Proceeds to a chi2 test. If the resulting p-value is lower
    than the statistical significance level, it computes and returns
    the associated phi value
    """
    # Drop missing values
    tmp = pd.DataFrame({"var1": var1, "var2": var2})
    tmp.dropna(axis=1)
    # Cross table building
    crosstab = np.array(pd.crosstab(tmp["var1"], tmp["var2"]))
    # Keeping of the test statistic of the Chi2 test
    chi2_stat, p_value, _, _ = chi2_contingency(crosstab, correction=False)
    n = crosstab.sum().sum()
    min_dim = min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)
    if p_value < alpha:
        return np.sqrt(chi2_stat / (n * min_dim))
    else:
        return np.nan


def corr_cramersV(df: pd.DataFrame, alpha: float = 0.05):
    """Build a triangular matrix of Phi coefficients for categorical variables."""
    categorical_columns = df.columns.values
    prod = itertools.product(categorical_columns, repeat=2)
    n = len(categorical_columns)
    matrix = np.full((n, n), np.nan)

    for index, item in enumerate(prod):
        phi = conditional_cramersV(df[item[0]], df[item[1]], alpha=alpha)
        matrix[index // n, index % n] = phi

    return pd.DataFrame(matrix, index=categorical_columns, columns=categorical_columns)


def calculate_cat_entropy(cat: pd.Series, normalized: bool = False) -> float:
    """This function aims at computing the entropy for
    a categorical variable
    """
    freqs = cat.value_counts() / cat.shape[0]
    entropy = np.sum(-freqs * np.log(freqs))
    if normalized:
        if len(freqs) > 1:
            entropy = entropy / np.log(len(freqs))
    return entropy


def compute_classifier_scores(
    classifier_name: List[str],
    scores: Dict,
    keys: List[str],
    confidence_level: float,
    pattern: str,
):
    """
    This functions computes and return the confidence intervals of several metrics
    for a given classifier
    """
    n_splits = len(scores[list(scores.keys())[0]])
    t_val = t.ppf((1 + confidence_level) / 2, n_splits - 1)
    res = dict()
    res["Classifier"] = classifier_name
    for k in keys:
        res[re.search(pattern, k).group(0)] = [
            np.round(np.mean(scores[k]) - t_val * sem(scores[k]), 3),
            np.round(np.mean(scores[k]) + t_val * sem(scores[k]), 3),
        ]
    return res


def compute_classifier_scores_b(
    classifier_names, scores, cis_to_return, confidence_level: float = 0.05
):
    """
    This functions computes and return the confidence intervals of several metrics
    for a given classifier
    """
    n_splits = len(scores[list(scores.keys())[0]])
    t_val = t.ppf((1 + confidence_level) / 2, n_splits - 1)
    res = dict()
    res["Classifier"] = classifier_names
    for it in cis_to_return:
        res[it] = [
            np.round(np.mean(scores[it]) - t_val * sem(scores[it]), 3),
            np.round(np.mean(scores[it]) + t_val * sem(scores[it]), 3),
        ]
    return res
