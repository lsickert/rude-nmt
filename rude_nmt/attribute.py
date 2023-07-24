"""this module contains the feature-attribution functions"""
import inseq
from inseq.data.aggregator import (
    AggregatorPipeline,
    SequenceAttributionAggregator,
    SubwordAggregator,
)
from datasets import Dataset
from spacy.training import Alignment
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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
    """attribute the dataset using the given attribution_method

    Args:
        ds (Dataset): the dataset
        src_lang (str): the source language
        trg_lang (str): the target language
        attribution_method (str): the attribution method to use
        batch_size (int, optional): the batch size. Defaults to 32.
        attribute_target (bool, optional): whether to attribute the target. Defaults to False.
        force_regen (bool, optional): whether to force the regeneration of the dataset. Defaults to False.

    Returns:
        Dataset: the attributed dataset
    """

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

    ds = ds.map(
        attribute_samples,
        batched=True,
        batch_size=1000,
        load_from_cache_file=not force_regen,
        fn_kwargs={
            "model": model,
            "aggregator": aggregator,
            "src_lang": src_lang,
            "trg_lang": trg_lang,
            "batch_size": batch_size,
            "attribute_target": attribute_target,
        },
    )

    return ds


def attribute_samples(
    samples: dict,
    model: inseq.AttributionModel,
    aggregator: AggregatorPipeline,
    src_lang: str,
    trg_lang: str,
    batch_size: int,
    attribute_target: bool = False,
) -> dict:
    """attribute a batch of samples using the given model and aggregator

    Args:
        samples (dict): the samples
        model (inseq.AttributionModel): the model
        aggregator (AggregatorPipeline): the aggregator
        src_lang (str): the source language
        trg_lang (str): the target language
        batch_size (int): the batch size
        attribute_target (bool, optional): whether to attribute the target. Defaults to False.

    Returns:
        dict: the attributed samples
    """

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

    samples[f"{trg_lang}_cross_attributions"] = src_attributions
    samples[f"{trg_lang}_step_scores"] = step_scores

    if attribute_target:
        samples[f"{trg_lang}_trg_attributions"] = trg_attributions

    return samples


def align_tokenizations_to_spacy(
    example: dict, split_col: str, spacy_col: str, alignment_col: str
):
    """align the feature-attribution tokenizations (whitespace-split) to the spacy tokenizations"""
    alignment = Alignment.from_strings(example[split_col].split(), example[spacy_col])
    aligned = []
    align_idx = 0
    for i in alignment.x2y.lengths:
        next_idx = align_idx + i
        aligned.append(example[alignment_col][alignment.x2y.data[align_idx]])
        align_idx = next_idx
    return aligned


def map_align_tokenizations(
    example: dict, split_col: str, spacy_col: str, alignment_col: str
):
    """map the feature-attribution tokenizations (whitespace-split) to the spacy tokenizations for a single example"""
    example[f"{alignment_col}_aligned"] = align_tokenizations_to_spacy(
        example, split_col, spacy_col, alignment_col
    )
    return example


def get_avg_pos_attr(
    ds: Dataset,
    split_col_src: str,
    spacy_col_src: str,
    pos_col_src: str,
    split_col_trg: str,
    spacy_col_trg: str,
    form_map_trg: str,
    attribute_col: str,
):
    """get the average source sentence attribution per POS tag for the dataset

    Args:
        ds (Dataset): the dataset
        split_col_src (str): the source sentence column (will be split by whitespace)
        spacy_col_src (str): the source sentence column with spacy tokenization
        pos_col_src (str): the source sentence column POS tags
        split_col_trg (str): the target sentence column (will be split by whitespace)
        spacy_col_trg (str): the target sentence column with spacy tokenizations
        form_map_trg (str): the target sentence formality map
        attribute_col (str): the column with the feature-attribution values

    Returns:
        dict: the average attribution per POS tag
    """
    all_pos = {}
    form_tokens_per_sent = []
    with tqdm(total=len(ds), desc="getting avg pos attr") as pbar:
        for example in ds:
            form_map_aligned = align_tokenizations_to_spacy(
                example, split_col_trg, spacy_col_trg, form_map_trg
            )
            form_tokens_per_sent.append(np.sum(form_map_aligned).astype(int))
            aligned_pos = align_tokenizations_to_spacy(
                example, split_col_src, spacy_col_src, pos_col_src
            )
            attr_form_filtered = [
                example[attribute_col][idx]
                for idx, f in enumerate(form_map_aligned)
                if f != 0
            ]

            if len(attr_form_filtered) == 0:
                pbar.update(1)
                continue

            if len(attr_form_filtered[0]) != len(aligned_pos):
                pbar.update(1)
                continue

            for attr in attr_form_filtered:
                df = pd.DataFrame({"pos": aligned_pos, "attr": attr})
                df = df.groupby("pos").max()
                for idx, row in df.iterrows():
                    if idx not in all_pos:
                        all_pos[idx] = []
                    all_pos[idx].append(row["attr"])
            pbar.update(1)

    num_formality_tokens = np.sum(form_tokens_per_sent).astype(int)
    for key in all_pos:
        all_pos[key] = round(np.sum(all_pos[key]) / num_formality_tokens, 3)

    return all_pos


def perform_contrastive_attr(
    ds: Dataset,
    source_col: str,
    target_col: str,
    contrast_col: str,
    src_lang: str,
    trg_lang: str,
    attribution_method: str,
    force_regen: bool = False,
) -> Dataset:
    """perform contrastive attribution on the dataset

    Args:
        ds (Dataset): the dataset
        source_col (str): the source column
        target_col (str): the target column
        contrast_col (str): the contrastive column
        src_lang (str): the source language
        trg_lang (str): the target language
        attribution_method (str): the attribution method
        force_regen (bool, optional): whether to force the regeneration of the dataset. Defaults to False.

    Returns:
        Dataset: the dataset with contrastive attribution
    """

    model = inseq.load_model(
        "facebook/mbart-large-50-many-to-many-mmt",
        attribution_method,
        tokenizer_kwargs={
            "src_lang": LANG_TAG_MAP[src_lang],
            "tgt_lang": LANG_TAG_MAP[trg_lang],
        },
    )

    ds = ds.filter(
        _filter_sentences_for_contrast,
        fn_kwargs={
            "model": model,
            "target_col": target_col,
            "contrast_col": contrast_col,
        },
        load_from_cache_file=not force_regen,
    )

    ds = ds.map(
        _get_contrastive_attr,
        fn_kwargs={
            "model": model,
            "source_col": source_col,
            "target_col": target_col,
            "contrast_col": contrast_col,
        },
        load_from_cache_file=not force_regen,
    )

    return ds


def get_contr_pos_tag_attr(ds: Dataset, model: inseq.AttributionModel, src_col: str):
    """retrieve the attribution per POS tag for the contrastive dataset

    Args:
        ds (Dataset): the dataset
        model (inseq.AttributionModel): the model
        src_col (str): the source column

    Returns:
        dict: the attribution per POS tag
    """
    pos_attr = {}

    for example in ds:
        clean_src_tokens = [
            tok.replace("▁", " ")
            for tok in model.encode(example[src_col]).input_tokens[0][1:-1]
        ]

        aligned = Alignment.from_strings(
            clean_src_tokens, example[f"ws_tokens_{src_col}"]
        )

        contr_idx = 0

        for idx, prob in enumerate(example["contrastive_prob"]):
            if prob != 0.0:
                contr_idx = idx
                break

        clean_src_attr = example["contrastive_attr"][contr_idx][1:-1]

        aligned_idx = []
        for idx, i in enumerate(aligned.x2y.lengths):
            if i == 0:
                continue
            aligned_idx.append(idx)

        aligned_attr_list = []
        pos_list = []
        aligned_clean_src_tokens = []
        for idx, i in enumerate(aligned_idx):
            aligned_attr_list.append(clean_src_attr[i])
            aligned_clean_src_tokens.append(clean_src_tokens[i])
            pos_list.append(example[f"upos_tags_{src_col}"][aligned.x2y.data[idx]])

        df = pd.DataFrame({"pos": pos_list, "attr": aligned_attr_list})
        # df = df.groupby("pos").max()
        df = df.loc[df.groupby("pos")["attr"].idxmax()]

        for idx, row in df.iterrows():
            if row["pos"] not in pos_attr:
                pos_attr[row["pos"]] = []
            pos_attr[row["pos"]].append(row["attr"])

    for key in pos_attr:
        pos_attr[key] = round(np.sum(pos_attr[key]) / len(ds), 3)

    return pos_attr


def get_max_contr_pos_tags(ds: Dataset, model: inseq.AttributionModel, src_col: str):
    """retrieve the POS tags with the highest contrastive attribution per sample

    Args:
        ds (Dataset): the dataset
        model (inseq.AttributionModel): the model
        src_col (str): the source column

    Returns:
        list: the POS tags with the highest contrastive attribution
    """
    all_pos_list = []

    for example in ds:
        clean_src_tokens = [
            tok.replace("▁", " ")
            for tok in model.encode(example[src_col]).input_tokens[0][1:-1]
        ]

        aligned = Alignment.from_strings(
            clean_src_tokens, example[f"ws_tokens_{src_col}"]
        )

        contr_idx = 0

        for idx, prob in enumerate(example["contrastive_prob"]):
            if prob != 0.0:
                contr_idx = idx
                break

        clean_src_attr = example["contrastive_attr"][contr_idx][1:-1]

        max_attr_idx = np.argmax(np.abs(clean_src_attr))

        aligned_idx = []
        for idx, i in enumerate(aligned.x2y.lengths):
            if i == 0:
                continue
            aligned_idx.append(idx)

        for idx, i in enumerate(aligned_idx):
            if i == max_attr_idx:
                all_pos_list.append(
                    example[f"upos_tags_{src_col}"][aligned.x2y.data[idx]]
                )

    return all_pos_list


def _get_contrastive_attr(
    example: dict,
    model: inseq.AttributionModel,
    source_col: str,
    target_col: str,
    contrast_col: str,
) -> dict:
    """perform contrastive attribution on the given example

    Args:
        example (dict): the example
        model (inseq.AttributionModel): the model
        source_col (str): the source column
        target_col (str): the target column
        contrast_col (str): the contrastive column

    Returns:
        dict: the example with contrastive attribution
    """

    contrast = model.encode(example[contrast_col], as_targets=True)

    out = model.attribute(
        input_texts=example[source_col],
        generated_texts=example[target_col],
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff", "probability"],
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
        show_progress=False,
    )

    out.weight_attributions("contrast_prob_diff")

    example["contrastive_attr"] = out.sequence_attributions[
        0
    ].source_attributions.T.tolist()
    example["contrastive_prob"] = (
        out.sequence_attributions[0].step_scores["contrast_prob_diff"].tolist()
    )
    example["attr_prob"] = (
        out.sequence_attributions[0].step_scores["probability"].tolist()
    )

    return example


def _filter_sentences_for_contrast(
    example: dict, model: inseq.AttributionModel, target_col: str, contrast_col: str
) -> dict:
    """filter the dataset for sentences that are different between the target and contrastive column

    Args:
        example (dict): the example
        model (inseq.AttributionModel): the model
        target_col (str): the target column
        contrast_col (str): the contrastive column

    Returns:
        dict: the example
    """

    formal = model.encode(example[target_col], as_targets=True)
    informal = model.encode(example[contrast_col], as_targets=True)

    if len(formal.input_tokens[0]) == len(informal.input_tokens[0]):
        for i in range(len(formal.input_tokens[0])):
            if formal.input_tokens[0][i] != informal.input_tokens[0][i]:
                return True

    return False
