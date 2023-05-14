"""this module contains the feature-attribution functions"""
import inseq
from inseq.data.aggregator import (
    AggregatorPipeline,
    SequenceAttributionAggregator,
    SubwordAggregator,
)
from datasets import Dataset

from .translation import LANG_TAG_MAP, LANG_COL_MAP


def attribute_ds(
    ds: Dataset,
    src_lang: str,
    trg_lang: str,
    attribution_method: str,
    batch_size: int = 16,
    attribute_target: bool = False,
) -> Dataset:
    """attribute the dataset using the given attribution_method"""

    # for testing
    # ds = ds.select(range(64))

    # load the model
    model = inseq.load_model(
        "facebook/mbart-large-50-many-to-many-mmt",
        attribution_method,
        tokenizer_kwargs={
            "src_lang": LANG_TAG_MAP[src_lang],
            "tgt_lang": LANG_TAG_MAP[trg_lang],
        },
    )

    # create the aggregator
    aggregator = AggregatorPipeline([SubwordAggregator, SequenceAttributionAggregator])

    attributions = model.attribute(
        input_texts=ds[LANG_COL_MAP[src_lang]],
        generated_texts=ds[f"{trg_lang}_nmt"],
        attribute_target=attribute_target,
        batch_size=batch_size,
        step_scores=["probability"],
        pretty_progress=False,
        device=inseq.utils.get_default_device(),
    )

    attributions = attributions.aggregate(aggregator=aggregator)

    src_attributions = []
    step_scores = []
    trg_attributions = []

    for attr in attributions.sequence_attributions:
        src_attributions.append(attr.source_attributions[1:-1].T[1:-1].tolist())
        step_scores.append(attr.step_scores["probability"].tolist())
        if attribute_target:
            trg_attributions.append(attr.target_attributions[1:-1].T[1:-1].tolist())

    col_name = f"{trg_lang}_src_attributions"
    ds = ds.add_column(name=col_name, column=src_attributions)

    col_name = f"{trg_lang}_step_scores"
    ds = ds.add_column(name=col_name, column=step_scores)

    if attribute_target:
        col_name = f"{trg_lang}_trg_attributions"
        ds = ds.add_column(name=col_name, column=trg_attributions)

    return ds
