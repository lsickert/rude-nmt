"""provides functions to analyze and visualize the tatoeba dataset"""
import re
from typing import Optional
from datasets import Dataset

from . import preprocess


def get_one_word_sentences(
    dataset: Optional[Dataset] = None, lang: str = "target", split: str = "train"
) -> Dataset:
    """extract all single-word sentences in either the source or target language"""
    if dataset is None:
        dataset = preprocess.get_dataset()[split]

    return dataset.filter(lambda ex: not re.search(".+\s.+", ex[lang]), num_proc=8)

def get_formality_plot(ds: Dataset, form_col: str) -> None:
    """plot the distribution of formality labels in the dataset"""
    df = ds.to_pandas()
    df[form_col].astype('category')
    rows = len(df.index)
    ax = df[form_col].value_counts().plot(kind='bar')
    ax.set_ylabel("Number of Sentences")
    for p in ax.patches:
        b = p.get_bbox()
        ax.annotate(str(round(p.get_height()/rows * 100,2)), ((b.x0 + b.x1)/2 - 0.16, b.y1 + 8000))

    fig = ax.get_figure()
    fig.savefig(f"{form_col}.png", bbox_inches="tight")
