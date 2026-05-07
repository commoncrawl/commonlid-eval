"""Language-agnostic text cleaner ported from OpenLID-v2.

Source: https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/scripts/tools/openlid_normer.py
"""

from __future__ import annotations

import regex

_NONWORD_PAT = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACE_PAT = regex.compile(r"\s\s+")


def openlid_normer_clean_line(line: str) -> str:
    """Apply OpenLID-v2's language-agnostic line cleaner.

    The transformation order must be preserved exactly:
    strip whitespace, collapse newlines, lowercase, drop non-word + digit
    characters, then squeeze remaining runs of whitespace.
    """
    text = line.strip().replace("\n", " ").lower()
    text = regex.sub(_NONWORD_PAT, "", text)
    return regex.sub(_SPACE_PAT, " ", text)
