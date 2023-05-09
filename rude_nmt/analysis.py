"""provides analysis functions for the annotated datasets"""
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from d3blocks import D3Blocks


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
