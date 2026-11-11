"""
Censorship markers and span handling.

Every detector ends the same way: censored character spans in the ORIGINAL text, each
carrying the domain that flagged it.  Overlapping spans are merged, and each merged
segment is replaced by one marker naming every domain that flagged it, e.g.
``[[NAME/USERNAME]]``.  Output is rebuilt from spans, so every character outside a
censored segment comes back untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Domain key → tag.  The order here is the order tags appear in a combined marker.
DOMAIN_TAGS: dict[str, str] = {
    'name': 'NAME', 'username': 'USERNAME', 'email': 'EMAIL', 'telephone': 'TELEPHONE',
    'address': 'ADDRESS', 'organisation': 'ORGANISATION', 'id': 'ID',
    'password': 'PASSWORD', 'filepath': 'FILEPATH',
}
DOMAINS: tuple[str, ...] = tuple(DOMAIN_TAGS)
MARKER_OPEN, MARKER_CLOSE, MARKER_JOIN = '[[', ']]', '/'


def marker(domains: Iterable[str]) -> str:
    """``{'username', 'name'}`` → ``'[[NAME/USERNAME]]'``."""
    ordered = sorted(set(domains), key=DOMAINS.index)
    return MARKER_OPEN + MARKER_JOIN.join(DOMAIN_TAGS[d] for d in ordered) + MARKER_CLOSE


def clean_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace from both ends; drop spans with no letters or digits
    ('–' or '**' alone is never PII).  Returns None for a span to discard."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start and any(ch.isalnum() for ch in text[start:end]):
        return start, end
    return None


@dataclass
class Segment:
    """One merged censored run of the original text."""
    start: int
    end: int
    domains: set[str] = field(default_factory=set)


def merge_spans(spans: Iterable[tuple[int, int, str]]) -> list[Segment]:
    """Merge (start, end, domain) spans into non-overlapping segments: a span that
    overlaps or directly touches the previous segment extends it and adds its domain,
    so word pieces flagged one by one ('N', 'X', 'O') come out as a single marker."""
    segments: list[Segment] = []
    for start, end, domain in sorted(set(spans)):
        if segments and start <= segments[-1].end:
            segments[-1].end = max(segments[-1].end, end)
            segments[-1].domains.add(domain)
        else:
            segments.append(Segment(start, end, {domain}))
    return segments


def render(text: str, segments: list[Segment]) -> str:
    """Replace every segment of ``text`` with its marker."""
    out, pos = [], 0
    for seg in segments:
        out += [text[pos:seg.start], marker(seg.domains)]
        pos = seg.end
    return ''.join(out + [text[pos:]])
