import re

"""matches a capitalized inflection of `Sie` unless it occurs at the beginning of a sentence."""
FORMAL_RE = re.compile(r"\s(?:(Sie)|(Ihnen)|(Ihre)|(Ihren))\b")

"""matches an inflection of `du`."""
INFORMAL_RE = re.compile(r"\b[Dd](?:(u)|(ir)|(ein)|(eine)|(einen))\b")

def annotate_formality(example):
    """annotate the formality of a German sentence"""
    example = annotate_tv_formality(example)

    return example

def annotate_tv_formality(example):
    """
    annotate the formality of a German sentence by matching it through a regex 
    based on the existence of the formality indicators of du (informal) and Sie (formal).
    A value of 0 means no formality information was found, a value of 1 means informal, 2 formal and 3 is ambiguous
    """

    form = 0

    if INFORMAL_RE.search(example["source"]) is not None:
        form += 1

    if FORMAL_RE.search(example["source"]) is not None:
        form += 2
    
    example["de_formality"] = form

    return example
