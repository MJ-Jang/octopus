#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import unicodedata

# ==================================================


# Hangeul patterns
chosung = (
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

jungsung = (
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
)

jongsung = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)


def is_hangeul(one_character: chr) -> bool:
    """ Judge whether input character is hangeul or not """
    return 0xAC00 <= ord(one_character[:1]) <= 0xD7A3


def explode_hangeul(one_hangeul: chr):
    """ Decompose hangeul character to cho, jung, jong sung """
    a = one_hangeul[:1]
    if not is_hangeul(a):
        return False
    b = ord(a) - 0xAC00
    cho = b // (21 * 28)
    jung = b % (21 * 28) // 28
    jong = b % 28
    if jong == 0:
        return chosung[cho], jungsung[jung]
    else:
        return chosung[cho], jungsung[jung], jongsung[jong]


def join_hangeul(inputlist):
    """ Change decomposed character list to merged string """
    result = ""
    cho, jung, jong = 0, 0, 0
    inputlist.insert(0, "")
    while len(inputlist) > 1:
        if inputlist[-1] in jongsung:
            if inputlist[-2] in jungsung:
                jong = jongsung.index(inputlist.pop())
            else:
                result += inputlist.pop()
        elif inputlist[-1] in jungsung:
            if inputlist[-2] in chosung:
                jung = jungsung.index(inputlist.pop())
                cho = chosung.index(inputlist.pop())
                result += chr(0xAC00 + ((cho * 21) + jung) * 28 + jong)
                cho, jung, jong = 0, 0, 0
            else:
                result += inputlist.pop()
        else:
            result += inputlist.pop()
    else:
        return result[::-1]


def flat_hangeul(text: str) -> list:
    """ Decompose all characters in input sentence """
    result = [explode_hangeul(i) if is_hangeul(i) else (i,) for i in text]
    result = list(sum(result, ()))
    return result


def merge_flatted_hangeul(text_list: list) -> str:
    """ Restore decomposed text to composed text """
    return join_hangeul(list(text_list))


def normalize_unicode(text: str) -> str:
    """ Unicode normalization """
    text = flat_hangeul(text)
    text = merge_flatted_hangeul(text)
    return unicodedata.normalize("NFC", text)
