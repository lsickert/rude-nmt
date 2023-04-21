"""provides functions to annotate formality for German"""
import re
import os
from typing import Any
import spacy
from spacy.tokens import Doc
from datasets import Dataset

FORMAL_RE = re.compile(
    r"\s(?:(Sie)|(Ihr)|(Ihrer)|(Ihnen)|(Ihre)|(Ihren)|(Euch)|(Euer)|(Eure)|(Euren))\b"
)
"""matches any capitalized inflection of `Sie` unless it occurs at the beginning of a sentence."""

INFORMAL_RE = re.compile(r"\b[Dd](?:(u)|(ich)|(ir)|(ein)|(eine)|(einen)|(einer))\b")
"""matches any inflection of `du`."""


def annotate_ds(ds: Dataset, force_regen: bool = False) -> Dataset:
    """annotate the German formality of a dataset"""
    print("##### Annotating German POS tags #####")
    ds = ds.map(get_pos_tags, batched=True, load_from_cache_file=not force_regen)

    print("##### Annotating German formality #####")
    ds = ds.map(
        annotate_formality_single,
        load_from_cache_file=not force_regen,
        num_proc=os.cpu_count(),
    )

    old_cache = ds.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return ds


def annotate_formality_single(example: dict[str, Any]) -> dict[str, Any]:
    """annotate the formality of a German sentence"""
    example = annotate_tv_formality_single(example)

    return example


def annotate_tv_formality_single(example: dict[str, Any]) -> dict[str, Any]:
    """
    annotate the formality of a German sentence by matching it through a regex
    based on the existence of the formality indicators of du (informal) and Sie (formal).
    """

    form = None

    if INFORMAL_RE.search(example["source"]) is not None:
        form = "informal" if form is None else "ambiguous"

    if FORMAL_RE.search(example["source"]) is not None:
        form = "formal" if form is None else "ambiguous"

    if form is None:
        form = "underspecified"

    example["de_formality"] = form

    if "de_nmt" in example:
        form = None

        if INFORMAL_RE.search(example["nmt_de"]) is not None:
            form = "informal" if form is None else "ambiguous"

        if FORMAL_RE.search(example["nmt_de"]) is not None:
            form = "formal" if form is None else "ambiguous"

        if form is None:
            form = "underspecified"

        example["de_formality_nmt"] = form

    return example


def get_pos_tags(examples: dict[str, list]) -> dict[str, list]:
    """get the POS tags of a German sentence"""
    spacy.prefer_gpu()
    nlp = spacy.load("de_dep_news_trf", disable=["lemmatizer"])

    examples["de_upos_tags"] = []
    examples["de_pos_tags"] = []
    examples["de_ws_tokens"] = []
    examples["de_sent_ids"] = []

    for doc in nlp.pipe(examples["source"]):
        examples["de_upos_tags"].append([token.pos_ for token in doc])
        examples["de_pos_tags"].append([token.tag_ for token in doc])
        examples["de_ws_tokens"].append([token.text for token in doc])
        examples["de_sent_ids"].append(get_sent_id(doc))

    if "de_nmt" in examples:
        examples["de_upos_tags_nmt"] = []
        examples["de_pos_tags_nmt"] = []
        examples["de_ws_tokens_nmt"] = []
        examples["de_sent_ids_nmt"] = []

        for doc in nlp.pipe(examples["de_nmt"]):
            examples["de_upos_tags_nmt"].append([token.pos_ for token in doc])
            examples["de_pos_tags_nmt"].append([token.tag_ for token in doc])
            examples["de_ws_tokens_nmt"].append([token.text for token in doc])
            examples["de_sent_ids_nmt"].append(get_sent_id(doc))

    return examples


def get_sent_id(example: Doc) -> list:
    """get the sentence index for each token"""
    if example.has_annotation("SENT_START"):
        return [sent_id for sent_id, sent in enumerate(example.sents) for token in sent]
    else:
        return [0 for token in example]
