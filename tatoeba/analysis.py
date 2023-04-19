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
