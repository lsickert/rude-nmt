from datasets import Dataset


def get_formality_plot(ds: Dataset, form_col: str) -> None:
    """plot the distribution of formality labels in the dataset"""
    df = ds.to_pandas()
    df[form_col].astype("category")
    rows = len(df.index)
    ax = df[form_col].value_counts().plot(kind="bar")
    ax.set_ylabel("Number of Sentences")
    for p in ax.patches:
        b = p.get_bbox()
        ax.annotate(
            str(round(p.get_height() / rows * 100, 2)),
            ((b.x0 + b.x1) / 2 - 0.16, b.y1 + 8000),
        )

    fig = ax.get_figure()
    fig.savefig(f"{form_col}.png", bbox_inches="tight")
