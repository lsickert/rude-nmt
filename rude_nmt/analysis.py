"""provides analysis functions for the annotated datasets"""
from typing import Optional
from datasets import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from d3blocks import D3Blocks
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency

ko_order = [
    "hasoseoche",
    "hasipsioche",
    "haeyoche",
    "haoche",
    "hageche",
    "haerache",
    "haeche",
    "underspecified",
    "ambiguous",
]

de_order = [
    "formal",
    "informal",
    "underspecified",
    "ambiguous",
]


def plot_translation_metrics(
    ds: Dataset,
    metric_cols: list,
    labels: list,
    show: bool = True,
    plt_name: str = "translation_metrics",
) -> None:
    """plot the distribution of translation metrics in the dataset

    Args:
        ds (Dataset): the dataset
        metric_cols (list): the columns to plot
        labels (list): the labels for the columns
        show (bool, optional): whether to show the plot. Defaults to True.
        plt_name (str, optional): the name of the plot. Defaults to "translation_metrics".

    Returns:
        None
    """
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

    plt.savefig(f"./plots/{plt_name}.png", bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_sankey(
    ds: Dataset,
    source_col: str,
    target_col: str,
    language: str,
    plt_name: str = "sankey",
    show: bool = True,
) -> None:
    """plot a sankey of two dataset formality columns against each other.

    Args:
        ds (Dataset): the dataset
        source_col (str): the source column
        target_col (str): the target column
        language (str): the language of the columns. Defines the ordering of the formality levels.
        plt_name (str, optional): the name of the plot. Defaults to "sankey".
        show (bool, optional): whether to show the plot. Defaults to True.

    Returns:
        None"""
    df = ds.to_pandas()
    df = df[[source_col, target_col]]

    df = df.groupby([source_col, target_col], as_index=False).size()

    order = ko_order if language == "ko" else de_order

    df[source_col] = df[source_col].astype(pd.CategoricalDtype(order, ordered=True))
    df[target_col] = df[target_col].astype(pd.CategoricalDtype(order, ordered=True))

    df = df.sort_values(by=[source_col, target_col])

    df.rename(
        columns={source_col: "source", target_col: "target", "size": "weight"},
        inplace=True,
    )

    df = df.fillna(0)

    df.to_csv(f"./plots/{plt_name}.csv")

    # add a space to the end of the target column to avoid circular link errors because of same values as in the source column
    def add_suffix(val):
        return val + " "

    df["target"] = df["target"].apply(add_suffix)

    df = df.astype({"weight": "float64"})

    d3 = D3Blocks(chart="Sankey")

    d3.sankey(
        df,
        title=plt_name.capitalize(),
        filepath=f"./plots/{plt_name}.html",
        showfig=show,
        figsize=(800, 600),
    )


def get_classification_report(
    ds: Dataset, label_col: str, pred_col: str, output_dict: bool = True
):
    """get a classification report for the given dataset.
    Wrapper for sklearn.metrics.classification_report

    Args:
        ds (Dataset): the dataset
        label_col (str): the column containing the labels
        pred_col (str): the column containing the predictions
        output_dict (bool, optional): whether to output a dictionary. Defaults to True.

    Returns:
        dict: the classification report
    """

    y_true = ds[label_col]
    y_pred = ds[pred_col]

    return classification_report(
        y_true, y_pred, digits=2, output_dict=output_dict, zero_division=0
    )


def get_cramers_v(
    ds: Dataset, form_col: str, cross_col: str, exclude_vals: Optional[list] = None
):
    """get the Cramer's V value for the given two columns in the dataset

    Args:
        ds (Dataset): the dataset
        form_col (str): the column containing the formality labels
        cross_col (str): the column containing the cross-annotated labels
        exclude_vals (Optional[list], optional): values to exclude from the analysis. Defaults to None.

    Returns:
        float: the Cramer's V value
    """

    df = ds.to_pandas()
    df[form_col] = df[form_col].astype("category")
    df[cross_col] = df[cross_col].astype("category")
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


def interpolate_attributions(
    attr_list: list,
    scale_size: Optional[tuple] = None,
    scale_factor: Optional[float] = None,
    interpolation_mode: str = "bicubic",
) -> np.ndarray:
    """interpolate the given list of attention maps

    Args:
        attr_list (list): the list of attention maps
        scale_size (Optional[tuple], optional): the size (width x height) to scale the attention maps to. Defaults to the average width and height of all attention maps.
        scale_factor (Optional[float], optional): a factor to scale the interpolated heatmap by. Defaults to None.

    Returns:
        np.ndarray: the interpolated heatmap
    """

    if scale_size is None:
        x = np.ceil(np.mean([attr.size(0) for attr in attr_list])).astype(int)
        y = np.ceil(np.mean([attr.size(1) for attr in attr_list])).astype(int)
        scale_size = (x, y)

    attr_list = [attr.unsqueeze(0).unsqueeze(0) for attr in attr_list]
    scaled_attr = [
        torch.nn.functional.interpolate(
            attr, size=scale_size, mode=interpolation_mode, align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
        for attr in attr_list
    ]

    scaled = torch.stack(scaled_attr).mean(dim=0).numpy()

    if scale_factor is not None:
        scaled = scaled * scale_factor

    arr = np.flip(scaled, axis=0)

    return arr.tolist()


def create_interpolation_plot_single(
    arr: np.ndarray,
    plt_name: Optional[str] = "interpolation_plot",
    plt_title: Optional[str] = None,
    save: bool = False,
) -> None:
    """create a single interpolation plot from the given attention heatmap

    Args:
        arr (np.ndarray): the interpolated attention heatmap
        plt_name (Optional[str], optional): the name of the plot. Defaults to "interpolation_plot".
        plt_title (Optional[str], optional): the title of the plot. Defaults to None.
        save (bool, optional): whether to save the plot. Defaults to False.

    Returns:
        None
    """

    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(arr, vmin=0, vmax=1, cmap="Reds")

    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Attributed Tokens")
    if plt_title is not None:
        ax.set_title(plt_title)

    fig.colorbar(pcm, ax=ax, shrink=0.8, location="right", label="Attention Weight")

    # fig.tight_layout()

    if save:
        fig.savefig(plt_name, dpi=300, bbox_inches="tight")


def create_interpolation_plot_multi(
    plot_list: list,
    plt_name: Optional[str] = "interpolation_plot",
    save: bool = False,
) -> None:
    """create multiple interpolation plots from a list of attention heatmaps

    Args:
        plot_list (list): the list of interpolation plots to create
        plt_name (Optional[str], optional): the name of the plot. Defaults to "interpolation_plot".
        save (bool, optional): whether to save the plot. Defaults to False.

        Returns:
            None
    """

    fig, ax = plt.subplots(4, 3, figsize=(8, 8), layout="constrained")

    for col in range(3):
        for row in range(4):
            plot_id = row * 3 + col
            pcm = ax[row, col].pcolormesh(
                plot_list[plot_id]["merged_attributions"], vmin=0, vmax=1, cmap="Reds"
            )

            ax[row, col].set_xlabel("Generated Tokens")
            ax[row, col].set_ylabel("Attributed Tokens")
            ax[row, col].set_title(f"{plot_list[plot_id]['title']}")

    fig.colorbar(
        pcm, ax=ax[1:3, 2], shrink=0.4, location="right", label="Attention Weight"
    )

    # fig.tight_layout()

    if save:
        fig.savefig(plt_name)
