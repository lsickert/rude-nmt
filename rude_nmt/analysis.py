"""provides analysis functions for the annotated datasets"""
from typing import Optional
import math
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from d3blocks import D3Blocks
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency


def plot_translation_metrics(
    ds: Dataset,
    metric_cols: list,
    labels: list,
    show: bool = True,
    plt_name: str = "translation_metrics",
):
    """plot the distribution of translation metrics in the dataset"""
    plot_map = []

    for col in metric_cols:
        hist = np.histogram(ds[col], bins=np.arange(1, 101), density=True)
        hist = hist[0] * 100
        plot_map.append(hist)

    for plot in plot_map:
        plt.plot(plot)

    plt.ylim(0, 14)
    plt.xlim(0, 100)
    plt.legend(labels)
    plt.xlabel("Score")
    plt.ylabel("Percentage of sentences")

    plt.savefig(f"{plt_name}.png", bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_sankey(
    ds: Dataset,
    source_col: str,
    target_col: str,
    plt_name: str = "sankey",
    show: bool = True,
):
    """plot a sankey of two columns against each other"""
    df = ds.to_pandas()
    df[source_col].astype("category")
    df[target_col].astype("category")
    df = df[[source_col, target_col]]

    df = df.groupby([source_col, target_col], as_index=False).size()

    df.rename(
        columns={source_col: "source", target_col: "target", "size": "weight"},
        inplace=True,
    )

    df = df.fillna(0)

    # add a space to the end of the target column to avoid circular link errors because of same values as in the source column
    def add_suffix(val):
        return val + " "

    df["target"] = df["target"].apply(add_suffix)

    df = df.astype({"weight": "float64"})

    d3 = D3Blocks(chart="Sankey")

    d3.sankey(
        df,
        title=plt_name.capitalize(),
        filepath=f"./{plt_name}.html",
        showfig=show,
        figsize=(800, 600),
    )


def get_classification_report(
    ds: Dataset, label_col: str, pred_col: str, output_dict: bool = True
):
    """get a classification report for the given dataset"""

    y_true = ds[label_col]
    y_pred = ds[pred_col]

    return classification_report(
        y_true, y_pred, digits=2, output_dict=output_dict, zero_division=0
    )


def get_cramers_v(
    ds: Dataset, form_col: str, cross_col: str, exclude_vals: Optional[list] = None
):
    """get the Cramer's V value for the given dataset"""

    df = ds.to_pandas()
    df[form_col].astype("category")
    df[cross_col].astype("category")
    if exclude_vals is not None:
        df = df[~df[form_col].isin(exclude_vals)]
        df = df[~df[cross_col].isin(exclude_vals)]

    cross_form = pd.crosstab(df[form_col], df[cross_col])

    chi2 = chi2_contingency(cross_form)[0]

    n = np.sum(cross_form.values)
    phi2 = chi2 / n
    r, k = cross_form.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
