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
    batch_size: int = 32,
    attribute_target: bool = False,
    force_regen: bool = False,
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

    print(f"##### attributing model on {model.device} with method {attribution_method}")

    # create the aggregator
    aggregator = AggregatorPipeline([SubwordAggregator, SequenceAttributionAggregator])

    def attribute_samples(samples):

        attributions = model.attribute(
            input_texts=samples[LANG_COL_MAP[src_lang]],
            generated_texts=samples[f"{trg_lang}_nmt"],
            attribute_target=attribute_target,
            batch_size=batch_size,
            step_scores=["probability"],
            show_progress=False,
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

        samples[f"{trg_lang}_src_attributions"] = src_attributions
        samples[f"{trg_lang}_step_scores"] = step_scores

        if attribute_target:
            samples[f"{trg_lang}_trg_attributions"] = trg_attributions

        return samples

    ds = ds.map(
        attribute_samples,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=not force_regen,
    )

    return ds
