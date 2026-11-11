"""
Offline tests of the engine glue — chunking, views, thresholds, merging, markers, the
redact() contract and settings — with a stand-in model, so no weights are downloaded.
Detection quality is not tested here; that was benchmarked in pii-detector.
"""
import re
from pathlib import Path

import numpy as np
import pytest

import fusion_tape
from fusion_tape import engine as engine_module
from fusion_tape.engine import Engine
from fusion_tape.models import MODEL_DOMAINS, TokenScores
from fusion_tape.pipes import token_spans
from fusion_tape.settings import load_settings
from fusion_tape.spans import marker, merge_spans, render

NAMES = {'jane', 'doe', 'john', 'smith', 'nunez'}
VOCAB = {'the', 'a', 'and', 'for', 'reset', 'please', 'team', 'report', 'old', 'password', 'was',
         'her', 'is', 'in', 'call', 'on', 'today', 'logged', 'login', 'hi', 'from', 'maybe'}
D = MODEL_DOMAINS.index


# A name the models find is censored by two pipes: NAME reads the name labels, and USERNAME
# reads the same labels on the normalised view (which is the raw text when nothing is
# username-shaped).  Hence [[NAME/USERNAME]] — the benchmarked, recall-first behaviour.
NAME = '[[NAME/USERNAME]]'


class StubModel:
    """Word tokens; NAMES words get top label 'name'; 'maybe' gets P(name) = 0.3 only."""
    key, context_limit, domains = 'stub', 64, {'name'}

    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return len(re.findall(r'\w+|[^\w\s]', text)) + 2

    def is_single_piece(self, word):
        return word.strip().lower() in VOCAB

    def score(self, texts, batch_size=16):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            toks = list(re.finditer(r'[^\W_]+|[^\w\s]|_', text))
            prob = np.zeros((len(toks), len(MODEL_DOMAINS)), dtype=np.float32)
            top = np.full(len(toks), -1)
            for i, m in enumerate(toks):
                if m.group().lower() in NAMES:
                    prob[i, D('name')], top[i] = 0.9, D('name')
                elif m.group().lower() == 'maybe':
                    prob[i, D('name')] = 0.3
            out.append(TokenScores(
                starts=np.array([m.start() for m in toks], dtype=int),
                ends=np.array([m.end() for m in toks], dtype=int),
                top_type=np.where(top >= 0, 0, -1), top_begin=np.zeros(len(toks), dtype=bool),
                top_domain=top, domain_prob=prob))
        return out, [0.001] * len(texts)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr('fusion_tape.settings.script_dir', lambda: tmp_path)
    return load_settings(Path(__file__).resolve().parent.parent / 'settings.yaml')


@pytest.fixture
def stub_engine(settings, monkeypatch):
    eng = Engine(settings, models=[StubModel()])
    monkeypatch.setattr(engine_module, '_engine', eng)
    return eng


# ── Markers and merging ───────────────────────────────────────────────────────
def test_marker_combines_domains_in_fixed_order():
    assert marker({'username', 'name'}) == '[[NAME/USERNAME]]'
    assert marker({'email'}) == '[[EMAIL]]'


def test_overlapping_spans_share_one_marker():
    text = 'user john_smith here'
    segs = merge_spans({(5, 15, 'username'), (5, 9, 'name'), (10, 15, 'name')})
    assert render(text, segs) == 'user [[NAME/USERNAME]] here'


def test_touching_spans_merge_but_separated_ones_do_not():
    text = 'code KLMNXO, Jane Doe'
    segs = merge_spans({(8, 9, 'id'), (9, 10, 'id'), (10, 11, 'id'), (13, 17, 'name'), (18, 21, 'name')})
    assert render(text, segs) == 'code KLM[[ID]], [[NAME]] [[NAME]]'


# ── Thresholds ────────────────────────────────────────────────────────────────
def _scores(probs, tops):
    n = len(probs)
    prob = np.zeros((n, len(MODEL_DOMAINS)), dtype=np.float32)
    prob[:, D('name')] = probs
    top = np.array([D('name') if t else -1 for t in tops])
    return TokenScores(starts=np.arange(n) * 2, ends=np.arange(n) * 2 + 1,
                       top_type=np.where(top >= 0, 0, -1), top_begin=np.zeros(n, dtype=bool),
                       top_domain=top, domain_prob=prob)


def test_top_label_always_flags_even_above_threshold():
    sc = _scores([0.4, 0.1], [True, False])
    assert token_spans(sc, 'name', 0.99) == [(0, 1)]


def test_threshold_adds_tokens_and_lowering_it_adds_more():
    sc = _scores([0.9, 0.3, 0.05], [True, False, False])
    assert token_spans(sc, 'name', 0.5) == [(0, 1)]
    assert token_spans(sc, 'name', 0.25) == [(0, 3)]
    assert token_spans(sc, 'name', 0.01) == [(0, 5)]


def test_threshold_from_settings_reaches_the_pipe(stub_engine):
    assert fusion_tape.redact('maybe today')[0] == 'maybe today'
    object.__setattr__(stub_engine.settings, 'thresholds',
                       {**stub_engine.settings.thresholds, 'name': 0.2})
    assert fusion_tape.redact('maybe today')[0] == '[[NAME]] today'      # only the NAME pipe's threshold moved


# ── redact() contract ─────────────────────────────────────────────────────────
def test_single_string_returns_string_and_no_log(stub_engine):
    redacted, log = fusion_tape.redact('Call Jane today, mail ops@example.com')
    assert redacted == f'Call {NAME} today, mail [[EMAIL]]'
    assert log is None


def test_list_returns_list_and_log_tuples(stub_engine):
    texts = ['Hi John', '', 'report in C:\\Temp\\q3.xlsx']
    redacted, log = fusion_tape.redact(texts, log=True)
    assert redacted == [f'Hi {NAME}', '', 'report in [[FILEPATH]]']
    assert [entry[0] for entry in log] == texts
    assert log[0][1] == {'John': NAME}
    assert log[2][1] == {'C:\\Temp\\q3.xlsx': '[[FILEPATH]]'}
    assert all(isinstance(entry[2], float) and entry[2] >= 0 for entry in log)


def test_username_hiding_a_name_gets_both_markers(stub_engine):
    redacted, log = fusion_tape.redact('login from john_smith today', log=True)
    assert redacted == 'login from [[NAME/USERNAME]] today'
    assert log[0][1] == {'john_smith': '[[NAME/USERNAME]]'}


def test_only_censored_characters_change(stub_engine):
    text = 'Hi  team,\n\tplease   reset — for Jane.\n'
    redacted, _ = fusion_tape.redact(text)
    assert redacted == text.replace('Jane', NAME)


def test_rejects_non_strings(stub_engine):
    with pytest.raises(TypeError):
        fusion_tape.redact(['ok', 3])


# ── One model run per unique string ───────────────────────────────────────────
def test_each_string_scored_once_for_all_nine_pipes(stub_engine):
    fusion_tape.redact(['Call Jane today', 'Call Jane today', 'Hi John'])
    (strings,) = stub_engine.models[0].calls          # a single score() call for the whole batch
    assert len(strings) == len(set(strings))
    assert 'Call Jane today' in strings and 'Hi John' in strings


def test_long_text_is_chunked_and_pii_in_every_chunk_is_censored(stub_engine):
    text = ' '.join(f'Sentence {i} is about Jane and the report.' for i in range(60))
    assert len(stub_engine.chunk(text)) > 1
    redacted, _ = fusion_tape.redact(text)
    assert 'Jane' not in redacted and redacted.count(NAME) == 60


# ── Settings ──────────────────────────────────────────────────────────────────
def test_default_dirs_are_made_next_to_the_run_location(stub_engine):
    s = stub_engine.settings
    assert [d.name for d in (s.models_dir, s.log_dir, s.outputs_dir)] == ['models', 'log', 'outputs']
    assert s.models_dir.is_dir() and s.log_dir.is_dir() and s.outputs_dir.is_dir()
    assert (s.log_dir / 'fusion_tape.log').is_file()


def test_log_file_never_contains_input_text(stub_engine):
    fusion_tape.redact('Call Jane on jane@example.com', log=True)
    content = (stub_engine.settings.log_dir / 'fusion_tape.log').read_text()
    assert 'redact: 1 text(s)' in content
    assert 'Jane' not in content and 'example.com' not in content


def test_settings_file_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr('fusion_tape.settings.script_dir', lambda: tmp_path)
    cfg = tmp_path / 'custom.yaml'
    cfg.write_text('paths:\n  models: /srv/pii-models\n  outputs: out\n'
                   'thresholds:\n  name: 0.35\n')
    s = load_settings(cfg)
    assert s.models_dir == Path('/srv/pii-models')
    assert s.outputs_dir.name == 'out' and s.outputs_dir.is_absolute()
    assert s.thresholds['name'] == 0.35 and s.thresholds['password'] == 0.5


def test_bad_threshold_is_rejected(tmp_path):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text('thresholds:\n  name: 1.5\n')
    with pytest.raises(ValueError):
        load_settings(cfg)
