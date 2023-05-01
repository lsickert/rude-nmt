"""provides functions to analyze and visualize the tatoeba dataset"""
import re
from typing import Optional
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
    ds: Dataset, form_col: str, exclude_vals: Optional[list] = None, ax_annotate_vals: tuple = (0.16, 8000)
) -> None:
    """plot the distribution of formality labels in the dataset"""
    df = ds.to_pandas()
    df[form_col].astype("category")
    if exclude_vals is not None:
        df = df[~df[form_col].isin(exclude_vals)]
    rows = len(df.index)
    ax = df[form_col].value_counts().plot(kind="bar")
    ax.set_ylabel("Number of Sentences")
    for p in ax.patches:
        b = p.get_bbox()
        ax.annotate(
            f"{round(p.get_height() / rows * 100, 2)}%",
            ((b.x0 + b.x1) / 2 - ax_annotate_vals[0], b.y1 + ax_annotate_vals[1]),
        )

    fig = ax.get_figure()
    fig.savefig(f"{form_col}.png", bbox_inches="tight")


def get_cross_formality_plot(
    ds: Dataset,
    form_col: str,
    cross_col: str,
    exclude_vals: Optional[list] = None,
    form_col_desc: str = None,
    cross_col_desc: str = None,
) -> None:
    """plot the cross-distribution of formality labels in the dataset"""
    df = ds.to_pandas()
    df[form_col].astype("category")
    df[cross_col].astype("category")
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
        for (proportion, y_loc) in zip(cross_form.loc[x], cross_form.loc[x].cumsum()):
            plt.text(
                x=(y_loc - proportion) + (proportion / 3.7),
                y=n - 0.475,
                s=f"{np.round(proportion*100, 1)}%",
            )
    plt.savefig("form_distr.png", bbox_inches="tight")
