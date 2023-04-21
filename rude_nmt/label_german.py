"""provides functions to annotate formality for German"""
import re
import spacy

FORMAL_RE = re.compile(
    r"\s(?:(Sie)|(Ihr)|(Ihrer)|(Ihnen)|(Ihre)|(Ihren)|(Euch)|(Euer)|(Eure)|(Euren))\b"
)
"""matches any capitalized inflection of `Sie` unless it occurs at the beginning of a sentence."""

INFORMAL_RE = re.compile(r"\b[Dd](?:(u)|(ich)|(ir)|(ein)|(eine)|(einen)|(einer))\b")
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


def get_pos_tags(examples):
    """get the POS tags of a German sentence"""
    spacy.prefer_gpu()
    nlp = spacy.load("de_dep_news_trf", disable=["parser", "lemmatizer"])

    examples["de_upos_tags"] = []
    examples["de_pos_tags"] = []
    examples["de_ws_tokens"] = []

    for doc in nlp.pipe(examples["source"]):
        examples["de_upos_tags"].append([token.pos_ for token in doc])
        examples["de_pos_tags"].append([token.tag_ for token in doc])
        examples["de_ws_tokens"].append([token.text for token in doc])

    return examples
