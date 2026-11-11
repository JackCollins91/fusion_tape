"""
Text rewrites that let the NAME models see names hidden in usernames (pii-detector evaluation §8b).

``normalise_for_names``  john_smith → john smith, JohnSmith → John Smith,
                         jane.doe92 → jane doe 92 (username-shaped tokens only)
``drop_first_letter``    xnunez → Nunez, jsmith92 → Smith (non-vocabulary words only);
                         if a NAME model labels the rewritten word, the WHOLE
                         original word is flagged

Both return an index map so every predicted span maps back onto the original
characters.
"""
from __future__ import annotations

import re
from typing import Callable

NAME_SPLIT_CHARS = '_.-'     # Mary-Jane → Mary Jane, jean-luc.picard → jean luc picard
SPLIT_LETTER_DIGIT = True    # Bernardo65 → Bernardo 65

# Only USERNAME-SHAPED tokens are rewritten: letters, optionally joined by '.', '_' or '-',
# with at most one trailing number of ≤ 4 digits, and a leading letter run of ≥ 2 Latin
# letters or any non-Latin letter.  Timestamps, UUIDs, hashes and wallets pass through.
_NORM_TOKEN_RE = re.compile(r'[^\W_]+(?:[._\-]+[^\W_]+)*')
_NAME_SHAPED_RE = re.compile(r'(?:[^\W\d_]{2,}|[^\W\d_A-Za-z])(?:[._\-]+[^\W\d_]+)*(?:[._\-]?\d{1,4})?')


def _name_shaped_mask(text: str) -> bytearray:
    mask = bytearray(len(text))
    for m in _NORM_TOKEN_RE.finditer(text):
        if _NAME_SHAPED_RE.fullmatch(m.group()):
            mask[m.start():m.end()] = b'\x01' * (m.end() - m.start())
    return mask


def normalise_for_names(text: str, split_chars: str = NAME_SPLIT_CHARS) -> tuple[str, list[int]]:
    """Return (normalised_text, idx_map); idx_map[k] = original index of output char k."""
    out: list[str] = []
    idx: list[int] = []
    n = len(text)
    shaped = _name_shaped_mask(text)
    for i, ch in enumerate(text):
        if not shaped[i]:
            out.append(ch); idx.append(i)
            continue
        prev = text[i - 1] if i > 0 else ''
        nxt = text[i + 1] if i + 1 < n else ''
        # 1) separators → space
        if ch in split_chars:
            if ch == '_' or (prev.isalnum() and nxt.isalnum()
                             and not (prev.isdigit() and nxt.isdigit())):
                out.append(' '); idx.append(i)
                continue
        # 2) camel-case boundary → insert a space before this capital
        if ch.isupper() and prev.isalnum():
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                out.append(' '); idx.append(i)
        # 3) letter → digit boundary: Bernardo65 → Bernardo 65
        elif SPLIT_LETTER_DIGIT and ch.isdigit() and prev.isalpha():
            out.append(' '); idx.append(i)
        out.append(ch); idx.append(i)
    return ''.join(out), idx


def map_span_back(idx_map: list[int], start: int, end: int) -> tuple[int, int]:
    """Map a [start, end) span in pre-processed text back to original offsets."""
    if end <= start or start >= len(idx_map):
        return start, end
    end = min(end, len(idx_map))
    return idx_map[start], idx_map[end - 1] + 1


# ── Drop-first-letter view ────────────────────────────────────────────────────
DROP_FIRST_MIN_LETTERS = 4                     # xnun → Nun is the shortest rewrite
_DROP_TOKEN_RE = re.compile(r'(?<![^\W_])([A-Za-z]{%d,})(\d{0,4})(?![^\W_])' % DROP_FIRST_MIN_LETTERS)


def drop_first_letter(text: str, is_vocab_word: Callable[[str], bool]
                      ) -> tuple[str, list[int], list[tuple[int, int]]]:
    """
    Return (rewritten_text, idx_map, rewritten_token_spans_in_original).
    ``is_vocab_word(w)`` must say whether w is an ordinary word (one piece in any
    loaded tokenizer); those, ALL-CAPS and camelCase tokens are left alone.
    """
    out: list[str] = []
    idx: list[int] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _DROP_TOKEN_RE.finditer(text):
        letters = m.group(1)
        if not (letters.islower() or (letters[0].isupper() and letters[1:].islower())):
            continue                                   # ALL-CAPS / camelCase: not this rule
        if is_vocab_word(letters):
            continue
        for i in range(pos, m.start()):
            out.append(text[i]); idx.append(i)
        rest = letters[1:]
        new = rest[0].upper() + rest[1:]
        for k, ch in enumerate(new):
            out.append(ch); idx.append(m.start() + 1 + k)
        spans.append((m.start(), m.end()))
        pos = m.end()                                  # trailing digits are dropped from the view
    for i in range(pos, len(text)):
        out.append(text[i]); idx.append(i)
    return ''.join(out), idx, spans
