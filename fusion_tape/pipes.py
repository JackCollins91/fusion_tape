"""
The nine domain pipes.

Each pipe is the detector combination that scored the highest recall for its domain
in the pii-detector benchmarks.  Every detector in a pipe is OR-ed: a span is
censored for the domain as soon as any one of them flags it.

Model detectors never run a model.  They read the ``TokenScores`` the engine already
computed — one forward pass per model per text view — and turn the probabilities into
a decision with the pipe's own threshold from ``settings.yaml``:

    flagged  =  top label is in the domain   OR   P(domain) >= threshold

Three views of a chunk exist, because names hide inside usernames:

    raw    the chunk as written
    norm   username-shaped tokens split up:   john_smith → john smith, JSmith → J Smith
    drop   first letter dropped from unknown words:  jsmith92 → Smith  (whole word flagged)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import rules
from .models import MODEL_DOMAINS, TokenScores
from .normalise import map_span_back
from .spans import clean_span

Span = tuple[int, int]


@dataclass(frozen=True)
class Model:
    """Both models' labels of ``labels`` read on one view of the chunk."""
    labels: str                 # a key of models.DOMAIN_LABELS
    view: str = 'raw'           # 'raw' | 'norm' | 'drop'


@dataclass(frozen=True)
class Rule:
    """A regex detector from rules.py."""
    detect: Callable[[str], list[dict]]


DOMAIN_PIPES: dict[str, tuple[Model | Rule, ...]] = {
    # F — regex only                                        benchmark recall 1.000, FPR 0.000
    'filepath':     (Rule(rules.naive_filepath_detect),),
    # S — username labels, NAME labels on the normalised and drop-first views, username rules
    'username':     (Model('username'), Model('name', 'norm'), Rule(rules.naive_username_detect),
                     Model('name', 'drop')),
    # I — models + structured-ID and crypto-wallet rules
    'id':           (Model('id'), Rule(rules.naive_uid_detect)),
    # O — models + legal-suffix / partnership rules
    'organisation': (Model('organisation'), Rule(rules.naive_org_detect)),
    # N — models on all three views + honorific / greeting rules
    'name':         (Model('name'), Model('name', 'norm'), Rule(rules.naive_name_detect),
                     Model('name', 'drop')),
    # T — models + worldwide phone regex with structural validation
    'telephone':    (Model('telephone'), Rule(rules.naive_phone_detect)),
    # A — models + street / secondary-unit rules
    'address':      (Model('address'), Rule(rules.naive_street_detect)),
    # P — models + symbol / random-token / secret-key rules
    'password':     (Model('password'), Rule(rules.naive_password_detect)),
    # E — regex only                                        benchmark recall 1.000, FPR 0.000
    'email':        (Rule(rules.naive_email_detect),),
}

VIEWS_USED: frozenset[str] = frozenset(
    d.view for pipe in DOMAIN_PIPES.values() for d in pipe if isinstance(d, Model))


@dataclass
class ChunkViews:
    """The texts a chunk is scored as.  ``pieces`` are (text, offset-in-view) pairs."""
    text: str
    norm_idx: list[int] | None = None                 # None → the norm view IS the raw text
    norm_pieces: tuple[tuple[str, int], ...] = ()
    drop_text: str | None = None                      # None → nothing to rewrite
    drop_idx: list[int] | None = None
    drop_tokens: tuple[Span, ...] = ()                # rewritten words, in chunk offsets

    def strings(self) -> list[str]:
        """Every string the models must score for this chunk."""
        out = [self.text] + [t for t, _ in self.norm_pieces]
        if self.drop_text is not None:
            out.append(self.drop_text)
        return out


def token_spans(scores: TokenScores, labels: str, threshold: float) -> list[Span]:
    """
    Character spans of the tokens flagged for ``labels``.  Neighbouring flagged tokens
    form one span; a span is split where the top label changes entity type or starts
    a new entity (B- tag), exactly as the benchmarked label grouping did.
    """
    if labels not in MODEL_DOMAINS or not len(scores):
        return []
    d = MODEL_DOMAINS.index(labels)
    top_in = scores.top_domain == d
    flagged = top_in | (scores.domain_prob[:, d] >= threshold)
    out: list[Span] = []
    prev = -2
    for i in np.flatnonzero(flagged).tolist():
        start, end = int(scores.starts[i]), int(scores.ends[i])
        if end <= start:
            continue
        joins = prev == i - 1 and out and not (
            top_in[i] and top_in[prev]
            and (scores.top_type[i] != scores.top_type[prev] or scores.top_begin[i]))
        if joins:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
        prev = i
    return out


def detect_chunk(views: ChunkViews, scores: dict[str, dict[str, TokenScores]],
                 thresholds: dict[str, float]) -> set[tuple[int, int, str]]:
    """Run all nine pipes on one chunk.  ``scores[model_key][string]`` holds the single
    forward pass per string; returns (start, end, domain) in chunk offsets."""
    text = views.text
    found: set[tuple[int, int, str]] = set()
    rule_cache: dict[Callable, list[Span]] = {}

    def add(domain: str, start: int, end: int) -> None:
        span = clean_span(text, start, end)
        if span:
            found.add((*span, domain))

    for domain, pipe in DOMAIN_PIPES.items():
        threshold = thresholds.get(domain, 0.5)
        for det in pipe:
            if isinstance(det, Rule):
                if det.detect not in rule_cache:
                    rule_cache[det.detect] = [(p['start'], p['end']) for p in det.detect(text)]
                for s, e in rule_cache[det.detect]:
                    add(domain, s, e)
                continue
            for by_string in scores.values():                      # every loaded model
                if det.view == 'raw' or (det.view == 'norm' and views.norm_idx is None):
                    for s, e in token_spans(by_string[text], det.labels, threshold):
                        add(domain, s, e)
                elif det.view == 'norm':
                    for piece, offset in views.norm_pieces:
                        for s, e in token_spans(by_string[piece], det.labels, threshold):
                            add(domain, *map_span_back(views.norm_idx, s + offset, e + offset))
                elif det.view == 'drop' and views.drop_text is not None:
                    for s, e in token_spans(by_string[views.drop_text], det.labels, threshold):
                        a, b = map_span_back(views.drop_idx, s, e)
                        for ts, te in views.drop_tokens:           # flag the WHOLE original word
                            if a < te and ts < b:
                                add(domain, ts, te)
    return found
