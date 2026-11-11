"""
Chunking — split, recombine, window, fit to context (pii-detector evaluation §3).

  Pass 1  split_sentences  – cut on sentence enders, remember what each cut was
  Pass 2  recombine        – undo cuts that are not real sentence ends
                             (abbreviations, initials, pieces shorter than √window)
  Pass 3  window_units     – pack units into ≤ max_chars windows that overlap by
                             ``overlap_units``; a unit that is itself too long is
                             hard-split TWICE with interleaved cut points, so an
                             entity cut by one tiling is whole in the other
  Pass 4  fit_to_context   – count tokens per chunk with every loaded tokenizer;
                             any chunk over the model context is re-chunked with
                             passes 1–3 at a smaller character budget, recursively

Ported from pii-detector, with its module-level settings collected into
``ChunkConfig``.  Pass 4 only runs when a token counter is given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from math import isqrt
from typing import Callable

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ChunkConfig:
    max_chunk_chars: int = 1_000          # character window for pass 3
    overlap_units: int = 2                # units shared by consecutive windows
    hard_split_snap_chars: int = 64       # a hard cut may move back this far to land on whitespace
    recombine_max_fraction: float = 0.5   # short-piece merges stop growing a unit at this fraction of the window
    max_refit_depth: int = 6              # recursion limit for the token-budget re-split
    min_refit_chars: int = 64             # never re-split below this many characters
    refit_safety: float = 0.9             # shrink the re-split budget a little below the estimate


# ── Pass 1: sentence enders ───────────────────────────────────────────────────
# Gap kinds: 'para' = blank line, 'excl' = after ! or ?, 'stop' = after '.' + Capital
_SENT_RE = re.compile(r'\n\s*\n|(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])')


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def split_sentences(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """Return (spans, gaps): gaps[i] is the boundary kind between spans[i] and spans[i+1]."""
    rank = {'stop': 0, 'excl': 1, 'para': 2}
    spans: list[tuple[int, int]] = []
    gaps: list[str] = []
    pos, pending = 0, None

    def _push(s, e):
        nonlocal pending
        s, e = _trim(text, s, e)
        if e > s:
            if spans:
                gaps.append(pending or 'stop')
            spans.append((s, e))
            pending = None

    for m in _SENT_RE.finditer(text):
        _push(pos, m.start())
        g = m.group()
        kind = ('para' if g.count('\n') >= 2
                else 'stop' if text[m.start() - 1] == '.' else 'excl')
        if pending is None or rank[kind] > rank[pending]:   # keep the strongest gap
            pending = kind
        pos = m.end()
    _push(pos, len(text))
    return spans, gaps


# ── Pass 2: recombination ─────────────────────────────────────────────────────
_ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'mx', 'dr', 'prof', 'sr', 'jr', 'sir', 'dame', 'rev', 'hon',
    'gen', 'col', 'capt', 'lt', 'sgt', 'maj', 'cmdr', 'gov', 'sen', 'rep', 'pres',
    'st', 'ave', 'rd', 'blvd', 'ln', 'ct', 'pl', 'cres', 'ter', 'tce', 'hwy', 'pkwy',
    'apt', 'ste', 'bldg', 'fl', 'no', 'nr', 'mt', 'ft', 'dept', 'univ',
    'inc', 'ltd', 'co', 'corp', 'llc', 'plc', 'bros',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
    'vs', 'etc', 'approx', 'fig', 'vol', 'ref', 'tel', 'ext', 'e.g', 'i.e', 'cf', 'al',
}
_LAST_TOKEN_RE = re.compile(r'([\w.]+)\.$')


def _ends_with_abbreviation(piece: str) -> bool:
    m = _LAST_TOKEN_RE.search(piece)
    if not m:
        return False
    tok = m.group(1).lower()
    return tok in _ABBREVIATIONS or (len(tok) == 1 and tok.isalpha())


def min_subsentence_chars(max_chars: int) -> int:
    return max(1, isqrt(max_chars))


def recombine(text: str, spans: list[tuple[int, int]], gaps: list[str],
              max_chars: int, min_len: int, cfg: ChunkConfig = ChunkConfig()) -> list[tuple[int, int]]:
    """
    A full-stop cut is undone when the left piece ends in a known abbreviation or
    a single initial ("Dr.", "St.", "Apt.", "J."), or when either piece is shorter
    than √(window).  Paragraph breaks and !/? are never undone.
    """
    if not spans:
        return []
    cap = int(max_chars * cfg.recombine_max_fraction)
    units = [spans[0]]
    for (s, e), gap in zip(spans[1:], gaps):
        cs, ce = units[-1]
        if gap == 'stop' and (
                _ends_with_abbreviation(text[cs:ce])          # always: "St. Lauraview", "Co. KGaA"
                or ((e - cs) <= cap and ((ce - cs) < min_len or (e - s) < min_len))):
            units[-1] = (cs, e)
        else:
            units.append((s, e))
    return units


# ── Pass 3: windows + double hard split ───────────────────────────────────────
def _snap(max_chars: int, cfg: ChunkConfig) -> int:
    return min(cfg.hard_split_snap_chars, max(1, max_chars // 8))


def _snap_back(text: str, target: int, floor: int, snap: int) -> int:
    lo = max(floor + 1, target - snap)
    for i in range(target, lo - 1, -1):
        if text[i - 1].isspace():
            return i
    return target


def _tile(text, start, end, first_target, stride, snap) -> list[int]:
    cuts, target, prev = [], first_target, start
    while target < end:
        cut = _snap_back(text, target, prev, snap)
        cut = target if cut <= prev else cut
        cuts.append(cut)
        prev, target = cut, target + stride
    return cuts


def _hard_split(text: str, start: int, end: int, max_chars: int, cfg: ChunkConfig) -> list[dict]:
    """Tile an over-long unit twice: cuts at start + k·S and at start + S/2 + k·S."""
    snap = _snap(max_chars, cfg)
    stride = max_chars - snap
    chunks = []
    for name, first in (('hard-A', start + stride), ('hard-B', start + stride // 2)):
        bounds = [start, *_tile(text, start, end, first, stride, snap), end]
        for a, b in zip(bounds, bounds[1:]):
            s, e = _trim(text, a, b)
            if e > s:
                chunks.append({'text': text[s:e], 'start': s, 'end': e, 'split': name})
    return chunks


def _density_window(text: str, max_chars: int, count_tokens: TokenCounter | None,
                    token_limit: int | None, cfg: ChunkConfig) -> int:
    """Character window holding about token_limit tokens of THIS text (logs/CSV/JSON are dense)."""
    if count_tokens is None or not token_limit or not text:
        return max_chars
    n = count_tokens(text)
    if n <= token_limit:
        return max_chars
    fit = int(len(text) * token_limit / n * cfg.refit_safety)
    return min(max_chars, max(cfg.min_refit_chars, fit))


def window_units(text: str, units: list[tuple[int, int]], max_chars: int, cfg: ChunkConfig,
                 count_tokens: TokenCounter | None = None, token_limit: int | None = None) -> list[dict]:
    chunks, i = [], 0
    while i < len(units):
        s0, e0 = units[i]
        if e0 - s0 > max_chars:
            size = _density_window(text[s0:e0], max_chars, count_tokens, token_limit, cfg)
            chunks.extend(_hard_split(text, s0, e0, size, cfg))
            i += 1
            continue
        j = i + 1
        while (j < len(units) and units[j][1] - s0 <= max_chars
               and units[j][1] - units[j][0] <= max_chars):
            j += 1
        chunks.append({'text': text[s0:units[j - 1][1]], 'start': s0,
                       'end': units[j - 1][1], 'split': 'sentence'})
        if j >= len(units):
            break
        i = j if units[j][1] - units[j][0] > max_chars else max(i + 1, j - cfg.overlap_units)
    return chunks


def chunk_text(text: str, max_chars: int, min_len: int, cfg: ChunkConfig,
               count_tokens: TokenCounter | None = None, token_limit: int | None = None) -> list[dict]:
    """Passes 1–3 on one string, offsets relative to that string."""
    if len(text) <= max_chars:
        return [{'text': text, 'start': 0, 'end': len(text), 'split': 'whole'}]
    spans, gaps = split_sentences(text)
    units = recombine(text, spans, gaps, max_chars, min_len, cfg) or [(0, len(text))]
    return window_units(text, units, max_chars, cfg, count_tokens, token_limit)


# ── Pass 4: token budget, recursive re-split ──────────────────────────────────
def fit_to_context(chunks: list[dict], count_tokens: TokenCounter, token_limit: int,
                   min_len: int, cfg: ChunkConfig, depth: int = 0) -> list[dict]:
    """Re-chunk any chunk over token_limit at len × limit / tokens × safety, recursively."""
    out = []
    for c in chunks:
        n = count_tokens(c['text'])
        if n <= token_limit:
            out.append({**c, 'tokens': n})
            continue
        if depth >= cfg.max_refit_depth or len(c['text']) <= cfg.min_refit_chars:
            out.append({**c, 'tokens': n})                  # hard stop for pathological input
            continue
        budget = max(cfg.min_refit_chars,
                     min(len(c['text']) - 1, int(len(c['text']) * token_limit / n * cfg.refit_safety)))
        subs = [{**s, 'start': s['start'] + c['start'], 'end': s['end'] + c['start'],
                 'split': f"{c['split']}>{s['split']}"}
                for s in chunk_text(c['text'], budget, min_len, cfg, count_tokens, token_limit)]
        out.extend(fit_to_context(subs, count_tokens, token_limit, min_len, cfg, depth + 1))
    return out


def chunk_document(text: str, cfg: ChunkConfig = ChunkConfig(),
                   count_tokens: TokenCounter | None = None, token_limit: int | None = None,
                   max_chars: int | None = None) -> list[dict]:
    """
    All passes.  Returns [{text, start, end, split, tokens}] with offsets into ``text``.
    ``split`` is 'whole' | 'sentence' | 'hard-A' | 'hard-B', or a '>'-joined path such
    as 'sentence>hard-B' when the token pass re-split a chunk.  Without a token
    counter pass 4 is skipped and ``tokens`` is None.

    The short-subsentence threshold is fixed by the TOP-LEVEL window (√max_chars)
    and kept at every refit level, so a re-split never undoes a pass-2 merge.
    """
    max_chars = max_chars or cfg.max_chunk_chars
    min_len = min_subsentence_chars(max_chars)
    chunks = chunk_text(text, max_chars, min_len, cfg, count_tokens, token_limit)
    if count_tokens is None or not token_limit:
        return [{**c, 'tokens': None} for c in chunks]
    return fit_to_context(chunks, count_tokens, token_limit, min_len, cfg)
