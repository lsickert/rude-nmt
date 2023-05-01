"""provides functions to annotate formality for Korean"""
import re
import os
from math import floor
from typing import Optional, Tuple, Any
import spacy
from spacy.tokens import Doc
from jamo import j2hcj
from datasets import Dataset

# start and end values of the unicode block containing all korean syllables
U_HAN_START = 0xAC00
U_HAN_END = 0xD7A3

KO_JA_INITIALS = [chr(code) for code in range(0x1100, 0x1113)]
"""unicode range containing all valid korean syllable initial consonants as individual characters"""

KO_JA_MEDIAL = [chr(code) for code in range(0x1161, 0x1176)]
"""unicode range containing all valid korean syllable medial vowels as individual characters"""

KO_JA_FINAL = [chr(code) for code in range(0x11A8, 0x11C3)]
"""unicode range containing all valid korean syllable final consonants as individual characters"""
# insert the possibility for no final consonant at the start
KO_JA_FINAL.insert(0, None)

# offset for syllable-to-character calculation
U_KO_OFFSET = 44032
U_KO_INIT_OFFSET = 588
U_KO_MED_OFFSET = 28

HASOSEOCHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<decl1>나이다)|(?P<decl2>사옵나이다)|(?P<decl3>옵나이다)|(?P<declRem1>더니이다)|(?P<declRem2>더이다)|(?P<declConj1>사오리이다)|(?P<declConj2>사오리다)|(?P<declConj3>오리이다)|(?P<int1>사옵나이까)|(?P<int2>옵나이까)|(?P<int3>사옵니까)|(?P<int4>옵니까)|(?P<int5>사오니까)|(?P<int6>오니까)|(?P<intInd1>사오리이까)|(?P<intInd2>사오리까)|(?P<intInd3>오리이까)|(?P<intInd4>오리까)|(?P<intAct1>리이까)|(?P<intAct2>리까)|(?P<intRefl1>더니이까)|(?P<intRefl2>더이까)|(?P<imp1>옵소서)|(?P<imp2>소서)|(?P<prop1>사이다))\b(?!\s\w)"
)

HASIPSIOCHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<declInd1>니다)|(?P<declInd2>뎁쇼)|(?P<declInd3>옵니다)|(?P<declInd4>사옵니다)|(?P<declDesc>올시다)|(?P<cert>지요|죠)|(?P<intInd>니까)|(?P<intInt>리까)|(?P<imp>시오)|(?P<prop1>십시다)|(?P<prop3>시라)|(?P<req>시사))\b(?!\s\w)"
)

HAOCHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<decInd1>오)|(?P<decInd2>소)|(?P<decKno1>다오)|(?P<decKno2>라오)|(?P<decInt>리다)|(?P<decExp>디다)|(?P<imp2>우)|(?P<exc1>구려)|(?P<exc2>로구려))\b(?!\s\w)"
)

HAGECHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<decInt1>네)|(?P<decInt2>레)|(?P<decKno>다네)|(?P<decTho>세)|(?P<decCon>니)|(?P<imp>게)|(?P<intInd1>(?<!구)나)|(?P<intInd2>가)|(?P<intUns>런가)|(?P<intDen>쏜가)|(?P<prop2>세나)|(?P<exc2>로세))\b(?!\s\w)"
)

HAERACHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<decInd>다)|(?P<decKno1>단다)|(?P<decKno2>란다)|(?P<decCon>거니)|(?P<decFac>더라)|(?P<decUns>리라)|(?P<decInt>리로다)|(?P<decCert>렷다)|(?P<imp1>어라)|(?P<imp2>라)|(?P<imp3>도록)|(?P<intInd1>냐)|(?P<intInd2>니)|(?P<intUnl>랴)|(?P<intAcc>련)|(?P<intExp1>던)|(?P<intExp2>디)|(?P<intComp1>담)|(?P<intComp2>남)|(?P<intComp3>람)|(?P<intUns>고)|(?P<intConf>자면서)|(?P<prop>자)|(?P<propAdv>자꾸나)|(?P<propOrd1>렴)|(?P<propOrd2>려무나)|(?P<intent>마)|(?P<excImp1>구나)|(?P<excImp2>로구나)|(?P<excInd>군)|(?P<excAsk>다니)|(?P<excAdm>데라니)|(?P<excWorr1>라)|(?P<excWorr2>세라)|(?P<ExcEmph>라니까)|(?P<subj>진저))\b(?!\s\w)"
)

HAEYOCHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<decl1>요|뇨)|(?P<decl2>이에요)|(?P<decl3>예요)|(?P<imp2>세요)|(?P<imp3>시어요)|(?P<prop>시지요|시죠))\b(?!\s\w)"
)

HAECHE_RE = re.compile(
    r"\b(?P<stem>\w+)(?:(?P<declInd1>어|아|와|봐|워)|(?P<declInd2>야)|(?P<declConf1>지)|(?P<declConf2>다지)|(?P<declKno>라지)|(?P<declCaus1>거든)|(?P<declCaus2>거들랑)|(?P<declExp>데)|(?P<declAnsw>고)|(?P<declNoAlt>밖에)|(?P<intInd>까)|(?P<intGues>게)|(?P<intTho>레)|(?P<intExp>려나)|(?P<intOp>거나)|(?P<intConf1>다면서)|(?P<intConf2>라면서)|(?P<excImp>구먼)|(?P<excRes>더라니)|(?P<excAdm1>데라니)|(?P<excAdm2>사)|(?P<excSelf>로고)|(?P<prop>자니까)|(?P<objNeg>세말이지)|(?P<monAsk1>다니)|(?P<monAsk2>라니)|(?P<monAdm>걸)|(?P<monEmph1>다니까)|(?P<monEmph2>라니까)|(?P<monDen1>대)|(?P<monDen2>(?<!구)나))\b(?!\s\w)"
)

HANNAMUN_TAGS = re.compile(r"pvd|pvg|pad|paa|ef")
UPOS_TAGS = ["VERB", "ADJ"]


def annotate_ds(
    ds: Dataset, rem_ambig: bool = False, force_regen: bool = False
) -> Dataset:
    """annotate the Korean formality of a dataset"""
    print("##### Annotating Korean POS tags #####")
    ds = ds.map(
        get_pos_tags,
        batched=True,
        load_from_cache_file=not force_regen,
        fn_kwargs={"col": "target"},
    )
    if "ko_nmt" in ds.column_names:
        ds = ds.map(
            get_pos_tags,
            batched=True,
            load_from_cache_file=not force_regen,
            fn_kwargs={"col": "ko_nmt"},
        )

    print("##### Annotating Korean formality #####")
    ds = ds.map(
        annotate_formality_single,
        load_from_cache_file=not force_regen,
        num_proc=os.cpu_count(),
    )

    if rem_ambig:
        ds = ds.filter(
            lambda ex: ex["ko_formality"] != "ambiguous",
            num_proc=os.cpu_count(),
            load_from_cache_file=not force_regen,
        )

    old_cache = ds.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return ds


def annotate_formality_single(example: dict[str, Any]) -> dict[str, Any]:
    """
    annotate the formality of a Korean sentence by matching it through a regex
    based on the endings of the main verb at the end of the sentence.
    """

    form = None

    num_words = len(example["ws_tokens_target"]) - 1
    sent = -1

    for i in range(num_words, -1, -1):
        if (
            any(HANNAMUN_TAGS.findall(example["pos_tags_target"][i]))
            and sent != example["sent_ids_target"][i]
        ):
            sent = example["sent_ids_target"][i]

            if is_hasoseoche(example["ws_tokens_target"][i]):
                form = (
                    "hasoseoche"
                    if (form is None or form == "hasoseoche")
                    else "ambiguous"
                )

            if is_hasipsioche(example["ws_tokens_target"][i]):
                form = (
                    "hasipsioche"
                    if (form is None or form == "hasipsioche")
                    else "ambiguous"
                )

            if is_haoche(example["ws_tokens_target"][i]):
                form = "haoche" if (form is None or form == "haoche") else "ambiguous"

            if is_hageche(example["ws_tokens_target"][i]):
                form = "hageche" if (form is None or form == "hageche") else "ambiguous"

            if is_haerache(example["ws_tokens_target"][i]):
                form = (
                    "haerache" if (form is None or form == "haerache") else "ambiguous"
                )

            if is_haeyoche(example["ws_tokens_target"][i]):
                form = "haeyoche" if (form is None or form == "haeyoche") else "ambiguous"

            if is_haeche(example["ws_tokens_target"][i]):
                form = "haeche" if (form is None or form == "haeche") else "ambiguous"

            # we need this special handling for haeche adjectives and some haeyoche
            # because otherwise the pattern would match for a lot of other endings as well
            # therefore we are only checking for it if no other pattern has been found so far
            if form is None:
                if example["pos_tags_target"][i] == "paa+ef":
                    if re.search(r"\w(?:다)\b", example["ws_tokens_target"][i]):
                        form = "haeche"
                if re.search(r"\w(?:지요|죠)\b", example["ws_tokens_target"][i]):
                    form = "haeyoche"

    if form is None:
        form = "underspecified"

    example["ko_formality"] = form

    if "ko_nmt" in example:

        form = None

        num_words = len(example["ws_tokens_ko_nmt"]) - 1
        sent = -1

        for i in range(num_words, -1, -1):
            if (
                any(HANNAMUN_TAGS.findall(example["pos_tags_ko_nmt"][i]))
                and sent != example["sent_ids_ko_nmt"][i]
            ):
                sent = example["sent_ids_ko_nmt"][i]

            if is_hasoseoche(example["ws_tokens_ko_nmt"][i]):
                form = (
                    "hasoseoche"
                    if (form is None or form == "hasoseoche")
                    else "ambiguous"
                )

            if is_hasipsioche(example["ws_tokens_ko_nmt"][i]):
                form = (
                    "hasipsioche"
                    if (form is None or form == "hasipsioche")
                    else "ambiguous"
                )

            if is_haoche(example["ws_tokens_ko_nmt"][i]):
                form = "haoche" if (form is None or form == "haoche") else "ambiguous"

            if is_hageche(example["ws_tokens_ko_nmt"][i]):
                form = "hageche" if (form is None or form == "hageche") else "ambiguous"

            if is_haerache(example["ws_tokens_ko_nmt"][i]):
                form = (
                    "haerache" if (form is None or form == "haerache") else "ambiguous"
                )

            if is_haeyoche(example["ws_tokens_ko_nmt"][i]):
                form = "haeyoche" if (form is None or form == "haeyoche") else "ambiguous"

            if is_haeche(example["ws_tokens_ko_nmt"][i]):
                form = "haeche" if (form is None or form == "haeche") else "ambiguous"
            
            # we need this special handling for haeche adjectives and some haeyoche
            # because otherwise the pattern would match for a lot of other endings as well
            # therefore we are only checking for it if no other pattern has been found so far
            if form is None:
                if example["pos_tags_ko_nmt"][i] == "paa+ef":
                    if re.search(r"\w(?:다)\b", example["ws_tokens_ko_nmt"][i]):
                        form = "haeche"
                if re.search(r"\w(?:지요|죠)\b", example["ws_tokens_ko_nmt"][i]):
                    form = "haeyoche"

        if form is None:
            form = "underspecified"

        example["ko_formality_nmt"] = form

    return example


def is_hasoseoche(sent: str) -> bool:
    """check if an example sentence is in hasoseoche formality"""
    match = HASOSEOCHE_RE.search(sent)
    if match is not None:
        return True

    return False


def is_hasipsioche(sent: str) -> bool:
    """check if an example sentence is in hasipsioche formality"""
    match = HASIPSIOCHE_RE.search(sent)
    if match is not None:
        if match["declInd1"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        elif match["declInd2"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄴ":
                return True
            else:
                return False
        elif match["cert"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        elif match["intInd"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        elif match["imp"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        elif match["req"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        else:
            return True

    return False


def is_haoche(sent: str) -> bool:
    """check if an example sentence is in haoche formality"""
    match = HAOCHE_RE.search(sent)
    if match is not None:
        if match["decExp"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㅂ":
                return True
            else:
                return False
        else:
            return True

    return False


def is_hageche(sent: str) -> bool:
    """check if an example sentence is in hageche formality"""
    match = HAGECHE_RE.search(sent)
    if match is not None:
        if match["decInt2"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["decTho"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["intInd2"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄴ":
                return True
            else:
                return False
        if match["intUns"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["intDen"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        else:
            return True

    return False


def is_haerache(sent: str) -> bool:
    """check if an example sentence is in haerache formality"""
    match = HAERACHE_RE.search(sent)
    if match is not None:
        if match["decInd"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if (
                chars is not None
                and chars[2] is not None
                and (j2hcj(chars[2]) == "ㄴ" or j2hcj(chars[2]) == "ㅆ")
            ):
                return True
            else:
                return False
        if match["intUns"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄴ":
                return True
            else:
                return False
        if match["excAdm"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["excWorr1"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["excWorr2"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["subj"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        else:
            return True
    return False


def is_haeyoche(sent: str) -> bool:
    """check if an example sentence is in haeyoche formality"""
    match = HAEYOCHE_RE.search(sent)
    if match is not None:
        return True

    return False


def is_haeche(sent: str) -> bool:
    """check if an example sentence is in haeche formality"""
    match = HAECHE_RE.search(sent)
    if match is not None:
        if match["declNoAlt"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["intInd"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["intTho"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["intOp"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["excAdm1"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["excAdm2"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["objNeg"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        if match["monAdm"] is not None:
            chars = separate_syllable(match["stem"][-1])
            if chars is not None and chars[2] is not None and j2hcj(chars[2]) == "ㄹ":
                return True
            else:
                return False
        else:
            return True

    return False


# see e.g. https://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_in_Unicode on how hangul syllables and individual characters can be converted
def separate_syllable(char: str) -> Optional[Tuple[str, str, str]]:
    """separate a full korean syllable into its individual characters, otherwise return `None`"""
    if is_hangul(char):
        no_off = ord(char) - U_KO_OFFSET
        initial = no_off // U_KO_INIT_OFFSET
        medial = (no_off - initial * U_KO_INIT_OFFSET) // U_KO_MED_OFFSET
        final = floor((no_off - initial * U_KO_INIT_OFFSET) - medial * U_KO_MED_OFFSET)

        return (KO_JA_INITIALS[initial], KO_JA_MEDIAL[medial], KO_JA_FINAL[final])

    else:
        return None


def is_hangul(char: str) -> bool:
    """check if a character is within the unicode range for hangul characters"""
    return U_HAN_START <= ord(char) <= U_HAN_END


def get_pos_tags(examples: dict[str, list], col: str) -> dict[str, list]:
    """get the POS tags of a Korean sentence"""

    nlp = spacy.load("ko_core_news_lg", disable=["lemmatizer"])

    examples[f"upos_tags_{col}"] = []
    examples[f"pos_tags_{col}"] = []
    examples[f"ws_tokens_{col}"] = []
    examples[f"sent_ids_{col}"] = []

    for doc in nlp.pipe(examples[col]):
        examples[f"upos_tags_{col}"].append([token.pos_ for token in doc])
        examples[f"pos_tags_{col}"].append([token.tag_ for token in doc])
        examples[f"ws_tokens_{col}"].append([token.text for token in doc])
        examples[f"sent_ids_{col}"].append(get_sent_id(doc))

    return examples


def get_sent_id(example: Doc) -> list:
    """get the sentence index for each token"""
    if example.has_annotation("SENT_START"):
        return [sent_id for sent_id, sent in enumerate(example.sents) for token in sent]
    else:
        return [0 for token in example]
