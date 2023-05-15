"""provides functions to annotate formality for German"""
import re
import os
from typing import Any
import spacy
from spacy.tokens import Doc
from spacy.training import Alignment
from datasets import Dataset

FORMAL_RE = re.compile(
    r"\b(?:Sie|Ihr|Ihrer|Ihnen|Ihre|Ihren|Ihrem|Ihres|Euch|Euer|Eure|Euren|Eures)\b"
)
"""matches any capitalized inflection of `Sie` unless it occurs at the beginning of a sentence."""

INFORMAL_RE = re.compile(
    r"\b(?:[Dd](?:u|ich|ir|ein|eine|einen|einer|eines|einem|eins)|euch|euer|eure|euren|eures)\b"
)
"""matches any inflection of `du`."""


def annotate_ds(
    ds: Dataset, rem_ambig: bool = False, force_regen: bool = False
) -> Dataset:
    """annotate the German formality of a dataset"""
    print("##### Annotating German POS tags #####")
    ds = ds.map(
        get_pos_tags,
        batched=True,
        load_from_cache_file=not force_regen,
        fn_kwargs={"col": "source"},
    )
    if "de_nmt" in ds.column_names:
        ds = ds.map(
            get_pos_tags,
            batched=True,
            load_from_cache_file=not force_regen,
            fn_kwargs={"col": "de_nmt"},
        )

    print("##### Annotating German formality #####")
    ds = ds.map(
        annotate_formality_single,
        load_from_cache_file=not force_regen,
        num_proc=os.cpu_count(),
    )

    if rem_ambig:
        ds = ds.filter(
            lambda ex: ex["de_formality"] != "ambiguous",
            num_proc=os.cpu_count(),
            load_from_cache_file=not force_regen,
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

    num_words = len(example["ws_tokens_source"])
    form_map = [0] * num_words

    for i in range(num_words - 1, -1, -1):

        if INFORMAL_RE.search(example["ws_tokens_source"][i]) is not None:
            form = "informal" if (form is None or form == "informal") else "ambiguous"
            form_map[i] = 1

        not_sent_begin = (
            i > 0 and example["sent_ids_source"][i - 1] == example["sent_ids_source"][i]
        )
        if (
            FORMAL_RE.search(example["ws_tokens_source"][i]) is not None
            and not_sent_begin
        ):
            form = "formal" if (form is None or form == "formal") else "ambiguous"
            form_map[i] = 1

    if form is None:
        form = "underspecified"

    example["de_formality"] = form
    example["de_formality_map"] = form_map

    if "de_nmt" in example:
        form = None
        num_words = len(example["ws_tokens_de_nmt"])
        form_map = [0] * num_words

        for i in range(num_words - 1, -1, -1):

            if INFORMAL_RE.search(example["ws_tokens_de_nmt"][i]) is not None:
                form = (
                    "informal" if (form is None or form == "informal") else "ambiguous"
                )

            not_sent_begin = (
                i > 0
                and example["sent_ids_de_nmt"][i - 1] == example["sent_ids_de_nmt"][i]
            )
            if (
                FORMAL_RE.search(example["ws_tokens_de_nmt"][i]) is not None
                and not_sent_begin
            ):
                form = "formal" if (form is None or form == "formal") else "ambiguous"

        if form is None:
            form = "underspecified"

        example["de_formality_nmt"] = form
        example["de_formality_map_nmt"] = form_map

    return example


def get_pos_tags(examples: dict[str, list], col: str) -> dict[str, list]:
    """get the POS tags of a German sentence"""
    spacy.prefer_gpu()
    nlp = spacy.load("de_dep_news_trf", disable=["lemmatizer"])

    examples[f"upos_tags_{col}"] = []
    examples[f"pos_tags_{col}"] = []
    examples[f"ws_tokens_{col}"] = []
    examples[f"sent_ids_{col}"] = []

    if f"ws_form_map_{col}" in examples:
        examples[f"form_map_{col}"] = []

    for i, doc in enumerate(nlp.pipe(examples[col])):
        examples[f"upos_tags_{col}"].append([token.pos_ for token in doc])
        examples[f"pos_tags_{col}"].append([token.tag_ for token in doc])
        examples[f"ws_tokens_{col}"].append([token.text for token in doc])
        examples[f"sent_ids_{col}"].append(get_sent_id(doc))

        if f"ws_form_map_{col}" in examples and f"ws_{col}" in examples:
            alignment = Alignment.from_strings(
                examples[f"ws_{col}"][i], [token.text for token in doc]
            )
            examples[f"form_map_{col}"].append(
                [
                    examples[f"ws_form_map_{col}"][i][k]
                    if alignment.y2x.data[j - 1] != k
                    else 0
                    for j, k in enumerate(alignment.y2x.data)
                ]
            )

    if f"ws_form_map_{col}" in examples:
        del examples[f"ws_form_map_{col}"]
        del examples[f"ws_{col}"]

    return examples


def get_sent_id(example: Doc) -> list:
    """get the sentence index for each token"""
    if example.has_annotation("SENT_START"):
        return [sent_id for sent_id, sent in enumerate(example.sents) for token in sent]
    else:
        return [0 for token in example]
