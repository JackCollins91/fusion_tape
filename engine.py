"""
The redaction engine: chunk → score once → nine domain pipes → merge → render.

For a batch of texts the engine

  1. chunks every text (sentence windows with overlap; over-long runs are tiled twice
     with interleaved cut points, so PII cut by one tiling is whole in the other),
  2. builds the three views of every chunk and collects the UNIQUE strings to score,
  3. runs each model once over those strings, in length-sorted batches,
  4. lets all nine domain pipes read the resulting probabilities,
  5. maps spans back to text offsets, merges overlaps and writes the markers.
"""
from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

from .chunking import ChunkConfig, chunk_document
from .scorer import PIIModel, load_model
from .normalise import drop_first_letter, normalise_for_names
from .pipes import ChunkViews, detect_chunk, views_used
from .settings import Settings, load_settings
from .spans import marker, merge_spans, render

log = logging.getLogger('fusion_tape')
LOG_FILE = 'fusion_tape.log'

LogEntry = tuple[str, dict[str, str], float]      # (input text, {redacted string: marker}, ms)


def _attach_log_file(settings: Settings) -> None:
    path = settings.log_dir / LOG_FILE
    if any(getattr(h, 'baseFilename', None) == str(path) for h in log.handlers):
        return
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


class Engine:
    """Loaded models + settings.  Build once (see ``get_engine``) and reuse."""

    def __init__(self, settings: Settings | None = None, models: list[PIIModel] | None = None):
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        _attach_log_file(self.settings)
        log.info('settings: %s | models=%s log=%s outputs=%s | thresholds=%s',
                 self.settings.source or 'built-in defaults', self.settings.models_dir,
                 self.settings.log_dir, self.settings.outputs_dir, self.settings.thresholds)
        if models is None:
            if self.settings.num_threads:
                import torch
                torch.set_num_threads(self.settings.num_threads)
            models = [load_model(spec, self.settings.models_dir, self.settings.device)
                      for spec in self.settings.models]
        self.models = models
        self.chunk_config = ChunkConfig(max_chunk_chars=self.settings.max_chunk_chars,
                                        overlap_units=self.settings.overlap_units)
        # Token budget per chunk: the smallest model context, minus headroom for the
        # spaces the name normaliser inserts.
        self.context_tokens = min(m.context_limit for m in models) - self.settings.token_margin
        self.is_vocab_word = lru_cache(maxsize=200_000)(self._is_vocab_word)

    # ── Tokens, chunks, views ─────────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """Tokens incl. special tokens under the hungriest loaded tokenizer."""
        return max(m.count_tokens(text) for m in self.models)

    def _is_vocab_word(self, word: str) -> bool:
        return any(m.is_single_piece(word.lower()) for m in self.models)

    def chunk(self, text: str) -> list[dict]:
        return chunk_document(text, self.chunk_config, self.count_tokens, self.context_tokens)

    def views(self, chunk_text: str, needed: frozenset[str] = frozenset({'norm', 'drop'})) -> ChunkViews:
        views = ChunkViews(chunk_text)
        if 'norm' in needed:
            norm, idx = normalise_for_names(chunk_text)
            if norm != chunk_text:
                views.norm_idx = idx
                # Inserted spaces can push a chunk that fitted past the model context.
                if self.count_tokens(norm) > self.context_tokens:
                    views.norm_pieces = tuple((c['text'], c['start']) for c in chunk_document(
                        norm, self.chunk_config, self.count_tokens, self.context_tokens))
                else:
                    views.norm_pieces = ((norm, 0),)
        if 'drop' in needed:
            dropped, idx, tokens = drop_first_letter(chunk_text, self.is_vocab_word)
            if tokens:
                views.drop_text, views.drop_idx, views.drop_tokens = dropped, idx, tuple(tokens)
        return views

    # ── Redaction ─────────────────────────────────────────────────────────────
    def redact_many(self, texts: list[str],
                    skip: frozenset[str] = frozenset()) -> tuple[list[str], list[LogEntry]]:
        """Redact ``texts`` together, leaving out the domain pipes in ``skip``.
        Returns the redacted texts and one log entry each."""
        t_call = time.perf_counter()
        needed = views_used(skip)                    # empty → no pipe reads the models at all
        seconds = [0.0] * len(texts)
        chunks: list[list[dict]] = []
        views: dict[str, ChunkViews] = {}            # chunk text → its views (shared by duplicates)
        owners: dict[str, set[int]] = {}             # string to score → texts that need it

        for i, text in enumerate(texts):
            t0 = time.perf_counter()
            chunks.append(self.chunk(text) if text.strip() else [])
            for c in chunks[i]:
                if c['text'] not in views:
                    views[c['text']] = self.views(c['text'], needed)
                for s in views[c['text']].strings() if needed else ():
                    owners.setdefault(s, set()).add(i)
            seconds[i] += time.perf_counter() - t0

        # One forward pass per model per unique string; its time is shared by its owners.
        strings = list(owners)
        scores: dict[str, dict] = {}
        for model in self.models:
            scored, spent = model.score(strings, self.settings.batch_size) if strings else ([], [])
            scores[model.key] = dict(zip(strings, scored))
            for s, sec in zip(strings, spent):
                for i in owners[s]:
                    seconds[i] += sec / len(owners[s])

        redacted: list[str] = []
        entries: list[LogEntry] = []
        for i, text in enumerate(texts):
            t0 = time.perf_counter()
            spans = set()
            for c in chunks[i]:
                for s, e, domain in detect_chunk(views[c['text']], scores, self.settings.thresholds, skip):
                    spans.add((c['start'] + s, c['start'] + e, domain))
            segments = merge_spans(spans)
            redacted.append(render(text, segments))
            found = {text[g.start:g.end]: marker(g.domains) for g in segments}
            seconds[i] += time.perf_counter() - t0
            entries.append((text, found, round(seconds[i] * 1000, 3)))

        log.info('redact: %d text(s), %d chars, %d chunk(s), %d scored string(s) x %d model(s), '
                 '%d segment(s) censored, skipped: %s, %.0f ms',
                 len(texts), sum(map(len, texts)), sum(map(len, chunks)), len(strings),
                 len(self.models), sum(len(e[1]) for e in entries), ', '.join(sorted(skip)) or 'none',
                 (time.perf_counter() - t_call) * 1000)
        return redacted, entries


# ── The shared engine ─────────────────────────────────────────────────────────
_engine: Engine | None = None
_lock = threading.Lock()


def get_engine() -> Engine:
    """The shared engine; settings are read and models loaded on the first call."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = Engine()
        return _engine


def reset_engine() -> None:
    """Forget the shared engine, so the next call re-reads settings.yaml."""
    global _engine
    with _lock:
        _engine = None
