import re
from math import floor
from typing import Optional

# start and end values of the unicode block containing all korean syllables
U_HAN_START = 0xAC00
U_HAN_END = 0xD7A3

"""unicode range containing all valid korean syllable initial consonants as individual characters"""
KO_JA_INITIALS = [chr(code) for code in range(0x1100, 0x1113)]

"""unicode range containing all valid korean syllable medial vowels as individual characters"""
KO_JA_MEDIAL = [chr(code) for code in range(0x1161, 0x1176)]

"""unicode range containing all valid korean syllable final consonants as individual characters"""
KO_JA_FINAL = [chr(code) for code in range(0x11A8, 0x11C3)]
# insert the possibility for no final consonant at the start
KO_JA_FINAL.insert(0, None)

# offset for syllable-to-character calculation
U_KO_OFFSET = 44032
U_KO_INIT_OFFSET = 588
U_KO_MED_OFFSET = 28

HASOSEOCHE_RE = re.compile(r"\b(?P<stem>\w+)(?:(?P<decl1>사옵나이다)|(?P<decl2>옵나이다)|(?P<declRem1>더니이다)|(?P<declRem2>더이다)|(?P<>declConj1>사오리이다)|(?P<declConj2>사오리다)|(?P<declConj3>오리이다)|(?P<int1>사옵나이까)|(?P<int2>옵나이까)|(?P<int3>사옵니까)|(?P<int4>옵니까)|(?P<int5>사오니까)|(?P<int6>오니까)|(?P<intInd1>사오리이까)|(?P<intInd2>사오리까)|(?P<intInd3>오리이까)|(?P<intInd4>오리까)|(?P<intAct1>리이까)|(?P<intAct2>리까)|(?P<intRefl1>더니이까)|(?P<intRefl2>더이까)|(?P<imp1>옵소서)|(?P<imp2>소서)|(?P<prop1>사이다))\b")

HASIPSIOCHE_RE = re.compile(r"\b(?P<stem>\w+)(?:니다|뎁쇼|옵니다|사옵니다|올시다|지요|니까|리까|시오|십시다|시지요|시라|시사)\b")

HAOCHE_RE = re.compile(r"\b(?P<stem>\w+)(?:오|소|다오|라오|리다|디다|우|구려|로구려)\b")

HAGECHE_RE = re.compile(r"\b(?P<stem>\w+)(?:네|레|다네|세|니|게|나|가|런가|쏜가|세나|로세)\b")

HAERACHE_RE = re.compile(r"\b(?P<stem>\w+)(?:다|단다|란다|거니|더라|리라|리로다|렷다|어라|라|도록|냐|니|랴|련|던|디|담|남|람|고|자면서|자|자꾸나|렴|려무나|마|구나|로구나|군|다니|데라니|세라|라니까|진저)\b")

HAEYOCHE_RE = re.compile(r"\b(?P<stem>\w+)(?:(?P<decl>요)|이에요|예요|세요|시어요)\b")

HAECHE_RE = re.compile(r"\b(?P<stem>\w+)(?:야|지|다지|라지|거든|거들랑|데|고|밖에|까|게|레|려나|거나|다면서|라면서|구먼|더라니|데라니|사|로고|자니까|세말이지|다니|라니|걸|다니까|라니까|대|나)\b")

def annotate_formality(example):

    return example

def is_hasoseoche(example: str) -> bool:

    match = HASOSEOCHE_RE.search(example)
    if match is not None:
        return True
    
    return False

# see e.g. https://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_in_Unicode on how hangul syllables and individual characters can be converted
def separate_syllable(char: str) -> Optional[tuple]:
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
    return U_HAN_START <= ord(char) <= U_HAN_END
