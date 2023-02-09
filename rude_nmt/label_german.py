"""provides functions to annotate formality for German"""
import re


FORMAL_RE = re.compile(r"\s(?:(Sie)|(Ihnen)|(Ihre)|(Ihren))\b")
"""matches any capitalized inflection of `Sie` unless it occurs at the beginning of a sentence."""

INFORMAL_RE = re.compile(r"\b[Dd](?:(u)|(ir)|(ein)|(eine)|(einen))\b")
"""matches any inflection of `du`."""


def annotate_formality(example):
    """annotate the formality of a German sentence"""
    example = annotate_tv_formality(example)

    return example


def annotate_tv_formality(example):
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

    return example
