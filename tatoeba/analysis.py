"""provides functions to analyze and visualize the tatoeba dataset"""
import re
from typing import Optional, Union
from datasets import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import preprocess

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


def get_one_word_sentences(
    dataset: Optional[Dataset] = None, lang: str = "target", split: str = "train"
) -> Dataset:
    """extract all single-word sentences in either the source or target language

    Args:
        dataset (Optional[Dataset], optional): the dataset to extract from. Defaults to None.
        lang (str, optional): the language to extract from. Defaults to "target".
        split (str, optional): the split to extract from. Defaults to "train".

    Returns:
        Dataset: the dataset containing only single-word sentences
    """
    if dataset is None:
        dataset = preprocess.get_dataset()[split]

    return dataset.filter(lambda ex: not re.search(".+\s.+", ex[lang]), num_proc=8)


def get_formality_plot_combined(
    ds: tuple[Dataset, Dataset],
    form_col: tuple[Union[list[str], str], Union[list[str], str]],
    language: tuple[str, str],
    plt_name: Optional[str] = None,
    exclude_vals: Optional[list] = None,
    ax_annotate_vals: tuple[tuple, tuple] = ((0.16, 8000), (0.16, 8000)),
    fig_size: tuple = (8, 6),
    width: float = 0.8,
    fig_relation: list = [1, 1],
    col_titles: tuple[Optional[list], Optional[list]] = (None, None),
    save: bool = True,
    x_rotation: tuple[Optional[int], Optional[int]] = (90, 90),
    ax_annotated: tuple[bool, bool] = (True, True),
    colors: list[str] = ["tab:blue", "tab:orange"],
) -> None:
    """plot the distribution of formality labels in the dataset

    Args:
        ds (tuple[Dataset, Dataset]): the datasets to plot
        form_col (tuple[Union[list[str], str], Union[list[str], str]]): the columns to plot
        language (tuple[str, str]): the languages of the datasets
        plt_name (Optional[str], optional): the name of the plot. Defaults to None.
        exclude_vals (Optional[list], optional): the values to exclude. Defaults to None.
        ax_annotate_vals (tuple[tuple, tuple], optional): the values to annotate the axes with. Defaults to ((0.16, 8000), (0.16, 8000)).
        fig_size (tuple, optional): the size of the figure. Defaults to (8, 6).
        width (float, optional): the width of the bars. Defaults to 0.8.
        fig_relation (list, optional): the width ratio of the figures. Defaults to [1, 1].
        col_titles (tuple[Optional[list], Optional[list]], optional): the titles of the columns. Defaults to (None, None).
        save (bool, optional): whether to save the plot. Defaults to True.
        x_rotation (tuple[Optional[int], Optional[int]], optional): the rotation of the x-axis labels. Defaults to (90, 90).
        ax_annotated (tuple[bool, bool], optional): whether to annotate the axes. Defaults to (True, True).
        colors (list[str], optional): the colors of the bars. Defaults to ["tab:blue", "tab:orange"].
    """

    df1 = ds[0].to_pandas()
    df2 = ds[1].to_pandas()

    if isinstance(form_col[0], str):
        form_col[0] = [form_col[0]]

    if isinstance(form_col[1], str):
        form_col[1] = [form_col[1]]

    order1 = ko_order if language[0] == "ko" else de_order
    order2 = ko_order if language[1] == "ko" else de_order

    for col in form_col[0]:
        df1[col] = df1[col].astype(pd.CategoricalDtype(order1, ordered=True))

        if exclude_vals is not None:
            df1 = df1[~df1[col].isin(exclude_vals)]

    for col in form_col[1]:
        df2[col] = df2[col].astype(pd.CategoricalDtype(order2, ordered=True))

        if exclude_vals is not None:
            df2 = df2[~df2[col].isin(exclude_vals)]

    rows1 = len(df1.index)
    rows2 = len(df2.index)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=fig_size, gridspec_kw={"width_ratios": fig_relation}
    )

    ax1 = (
        df1[form_col[0]]
        .apply(pd.Series.value_counts)
        .loc[order1]
        .plot(kind="bar", ax=ax1, width=width, color=colors)
    )
    ax2 = (
        df2[form_col[1]]
        .apply(pd.Series.value_counts)
        .loc[order2]
        .plot(kind="bar", ax=ax2, width=width, color=colors)
    )

    ax1.set_ylabel("Number of Sentences")
    ax1.set_ylim(0, len(df1.index))

    ax2.set_ylabel("Number of Sentences")
    ax2.set_ylim(0, len(df2.index))

    if col_titles[0] is not None:
        ax1.legend(col_titles[0])

    if col_titles[1] is not None:
        ax2.legend(col_titles[1])

    if x_rotation[0] is not None:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=x_rotation[0])

    if x_rotation[1] is not None:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=x_rotation[1])

    if ax_annotated[0]:
        for p in ax1.patches:
            b = p.get_bbox()
            ax1.annotate(
                f"{round(p.get_height() / rows1 * 100, 2)}%",
                (
                    (b.x0 + b.x1) / 2 - ax_annotate_vals[0][0],
                    b.y1 + ax_annotate_vals[0][1],
                ),
            )

    if ax_annotated[1]:
        for p in ax2.patches:
            b = p.get_bbox()
            ax2.annotate(
                f"{round(p.get_height() / rows2 * 100, 2)}%",
                (
                    (b.x0 + b.x1) / 2 - ax_annotate_vals[1][0],
                    b.y1 + ax_annotate_vals[1][1],
                ),
            )

    fig.tight_layout()

    if save:
        if plt_name is None:
            plt_name = form_col
        fig.savefig(f"./plots/{plt_name}.png", bbox_inches="tight")


def get_formality_plot(
    ds: Dataset,
    form_col: Union[list[str], str],
    language: str,
    plt_name: Optional[str] = None,
    exclude_vals: Optional[list] = None,
    ax_annotate_vals: tuple = (0.16, 8000),
    fig_size: tuple = (8, 6),
    width: float = 0.8,
    col_titles: Optional[list] = None,
    save: bool = True,
    x_rotation: Optional[int] = 90,
    ax_annotated: bool = True,
) -> None:
    """plot the distribution of formality labels in the dataset

    Args:
        ds (Dataset): the dataset to plot
        form_col (Union[list[str], str]): the columns to plot
        language (str): the language of the dataset
        plt_name (Optional[str], optional): the name of the plot. Defaults to None.
        exclude_vals (Optional[list], optional): the values to exclude. Defaults to None.
        ax_annotate_vals (tuple, optional): the values to annotate the axes with. Defaults to (0.16, 8000).
        fig_size (tuple, optional): the size of the figure. Defaults to (8, 6).
        width (float, optional): the width of the bars. Defaults to 0.8.
        col_titles (Optional[list], optional): the titles of the columns. Defaults to None.
        save (bool, optional): whether to save the plot. Defaults to True.
        x_rotation (Optional[int], optional): the rotation of the x-axis labels. Defaults to 90.
        ax_annotated (bool, optional): whether to annotate the axes. Defaults to True.
    """
    df = ds.to_pandas()

    if isinstance(form_col, str):
        form_col = [form_col]

    order = ko_order if language == "ko" else de_order

    for col in form_col:
        df[col] = df[col].astype(pd.CategoricalDtype(order, ordered=True))

        if exclude_vals is not None:
            df = df[~df[col].isin(exclude_vals)]

    rows = len(df.index)
    ax = (
        df[form_col]
        .apply(pd.Series.value_counts)
        .loc[order]
        .plot(kind="bar", width=width, figsize=fig_size)
    )
    ax.set_ylabel("Number of Sentences")
    ax.set_ylim(0, len(df.index))

    if col_titles is not None:
        ax.legend(col_titles)

    if x_rotation is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)

    if ax_annotated:
        for p in ax.patches:
            b = p.get_bbox()
            ax.annotate(
                f"{round(p.get_height() / rows * 100, 2)}%",
                ((b.x0 + b.x1) / 2 - ax_annotate_vals[0], b.y1 + ax_annotate_vals[1]),
            )

    fig = ax.get_figure()
    if save:
        if plt_name is None:
            plt_name = form_col
        fig.savefig(f"./plots/{plt_name}.png", bbox_inches="tight")


def get_cross_formality_plot(
    ds: Dataset,
    form_col: str,
    cross_col: str,
    plt_name: Optional[str] = None,
    exclude_vals: Optional[list] = None,
    form_col_desc: str = None,
    cross_col_desc: str = None,
    plot_title: str = "form_distribution",
    label_x: float = 3.7,
    label_y: float = 0.475,
    save: bool = True,
    colors: list[str] = ["tab:blue", "tab:orange"],
) -> None:
    """plot the cross-distribution of formality labels in the dataset

    Args:
        ds (Dataset): the dataset to plot
        form_col (str): the column to plot
        cross_col (str): the column to crossplot with
        plt_name (Optional[str], optional): the name of the plot. Defaults to None.
        exclude_vals (Optional[list], optional): the values to exclude. Defaults to None.
        form_col_desc (str, optional): the description of the formality column. Defaults to None.
        cross_col_desc (str, optional): the description of the cross column. Defaults to None.
        plot_title (str, optional): the title of the plot. Defaults to "form_distribution".
        label_x (float, optional): the x position of the labels. Defaults to 3.7.
        label_y (float, optional): the y position of the labels. Defaults to 0.475.
        save (bool, optional): whether to save the plot. Defaults to True.
        colors (list[str], optional): the colors of the bars. Defaults to ["tab:blue", "tab:orange"].
    """
    df = ds.to_pandas()

    df[form_col] = df[form_col].astype(pd.CategoricalDtype(ko_order, ordered=True))
    df[cross_col] = df[cross_col].astype(pd.CategoricalDtype(de_order, ordered=True))
    if exclude_vals is not None:
        df = df[~df[form_col].isin(exclude_vals)]
        df = df[~df[cross_col].isin(exclude_vals)]

    cross_form = pd.crosstab(df[form_col], df[cross_col], normalize="index")

    cross_form = cross_form.sort_values(by=[form_col], ascending=False)

    cross_form.plot(kind="barh", stacked=True, color=colors)
    plt.xlabel("Percentage of sentences")
    if form_col_desc is not None:
        plt.ylabel(form_col_desc)

    if cross_col_desc is not None:
        plt.legend(title=cross_col_desc, loc="lower right")

    for n, x in enumerate([*cross_form.index.values]):
        previous_x_pos = -1
        for (proportion, x_loc) in zip(cross_form.loc[x], cross_form.loc[x].cumsum()):
            x_pos = round((x_loc - proportion) + (proportion / label_x), 1)
            if x_pos <= previous_x_pos:
                x_pos += 0.1
            previous_x_pos = x_pos
            prop_label = np.round(proportion * 100, 1)
            if prop_label != 0.0:
                plt.text(
                    x=x_pos,
                    y=n - label_y,
                    s=f"{prop_label}%",
                )
    if save:
        if plt_name is None:
            plt_name = plot_title
        plt.savefig(f"./plots/{plt_name}.png", bbox_inches="tight")
