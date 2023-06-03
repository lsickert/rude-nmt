"""provides functions to analyze and visualize the tatoeba dataset"""
import re
from typing import Optional, Union
from datasets import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import preprocess


def get_one_word_sentences(
    dataset: Optional[Dataset] = None, lang: str = "target", split: str = "train"
) -> Dataset:
    """extract all single-word sentences in either the source or target language"""
    if dataset is None:
        dataset = preprocess.get_dataset()[split]

    return dataset.filter(lambda ex: not re.search(".+\s.+", ex[lang]), num_proc=8)


def get_formality_plot(
    ds: Dataset,
    form_col: Union[list[str], str],
    plt_name: Optional[str] = None,
    exclude_vals: Optional[list] = None,
    ax_annotate_vals: tuple = (0.16, 8000),
    col_titles: Optional[list] = None,
    save: bool = True,
    horizontal_x: bool = False,
) -> None:
    """plot the distribution of formality labels in the dataset"""
    df = ds.to_pandas()

    if isinstance(form_col, str):
        form_col = [form_col]

    for col in form_col:
        df[col] = df[col].astype("category")

        if exclude_vals is not None:
            df = df[~df[col].isin(exclude_vals)]

    rows = len(df.index)
    ax = df[form_col].apply(pd.Series.value_counts).plot(kind="bar")
    ax.set_ylabel("Number of Sentences")
    ax.set_ylim(0, len(df.index))

    if col_titles is not None:
        ax.legend(col_titles)

    if horizontal_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

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
) -> None:
    """plot the cross-distribution of formality labels in the dataset"""
    df = ds.to_pandas()
    df[form_col] = df[form_col].astype("category")
    df[cross_col] = df[cross_col].astype("category")
    if exclude_vals is not None:
        df = df[~df[form_col].isin(exclude_vals)]
        df = df[~df[cross_col].isin(exclude_vals)]

    cross_form = pd.crosstab(df[form_col], df[cross_col], normalize="index")

    cross_form.plot(kind="barh", stacked=True)
    plt.xlabel("Percentage of sentences")
    if form_col_desc is not None:
        plt.ylabel(form_col_desc)

    if cross_col_desc is not None:
        plt.legend(title=cross_col_desc)

    for n, x in enumerate([*cross_form.index.values]):
        previous_x_pos = -1
        for (proportion, x_loc) in zip(cross_form.loc[x], cross_form.loc[x].cumsum()):
            x_pos = round((x_loc - proportion) + (proportion / label_x),1)
            if x_pos <= previous_x_pos:
                x_pos += 0.1
            previous_x_pos = x_pos
            prop_label = np.round(proportion*100, 1)
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
