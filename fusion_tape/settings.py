"""
Settings — read ``settings.yaml`` and resolve the models / log / outputs directories.

Lookup order for the file:
  1. the path in the ``FUSION_TAPE_SETTINGS`` environment variable,
  2. ``settings.yaml`` next to the running script,
  3. ``settings.yaml`` in the repository root (the one shipped with FUSION TAPE).
Anything missing from the file falls back to the defaults below.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

SETTINGS_ENV = 'FUSION_TAPE_SETTINGS'
SETTINGS_NAME = 'settings.yaml'

# Domains whose pipe reads model probabilities, i.e. the ones that have a threshold.
THRESHOLD_DOMAINS = ('name', 'username', 'telephone', 'address', 'organisation', 'id', 'password')

DEFAULTS: dict = {
    'paths': {'models': None, 'log': None, 'outputs': None},
    'models': {
        'bert_small': {'repo': 'gravitee-io/bert-small-pii-detection',
                       'folder': 'gravitee_bert_small_pii', 'revision': None},
        'ettin': {'repo': 'kalyan-ks/ettin-68m-nemotron-pii',
                  'folder': 'ettin_nemotron_pii', 'revision': None},
    },
    'runtime': {'device': 'cpu', 'batch_size': 16, 'num_threads': None},
    'chunking': {'max_chunk_chars': 1000, 'overlap_units': 2, 'token_margin': 16},
    'thresholds': {d: 0.5 for d in THRESHOLD_DOMAINS},
}


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # 'bert_small' | 'ettin'
    repo: str
    folder: str
    revision: str | None = None


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    log_dir: Path
    outputs_dir: Path
    models: tuple[ModelSpec, ...]
    device: str
    batch_size: int
    num_threads: int | None
    max_chunk_chars: int
    overlap_units: int
    token_margin: int
    thresholds: dict[str, float] = field(default_factory=dict)
    source: Path | None = None          # the settings file that was read, if any

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.log_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


def script_dir() -> Path:
    """Directory of the running script.  Falls back to the working directory when there
    is no script of your own: a notebook, a REPL, ``python -m …`` or a console command."""
    main = sys.modules.get('__main__')
    main_file = getattr(main, '__file__', None)
    if main_file and getattr(main, '__spec__', None) is None and str(main_file).endswith('.py'):
        return Path(main_file).resolve().parent
    return Path.cwd()


def find_settings_file() -> Path | None:
    env = os.environ.get(SETTINGS_ENV)
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{SETTINGS_ENV} points to {path}, which does not exist')
        return path
    for candidate in (script_dir() / SETTINGS_NAME,
                      Path(__file__).resolve().parent.parent / SETTINGS_NAME):
        if candidate.is_file():
            return candidate
    return None


def _merged(defaults: dict, given: dict | None) -> dict:
    out = dict(defaults)
    for key, value in (given or {}).items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            out[key] = _merged(defaults[key], value)
        else:
            out[key] = value
    return out


def _resolve_dir(value, base: Path, default_name: str) -> Path:
    if value in (None, ''):
        return base / default_name
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def load_settings(path: str | Path | None = None) -> Settings:
    """Read the settings file (see module docstring for the lookup order)."""
    file = Path(path).expanduser() if path else find_settings_file()
    raw: dict = {}
    if file is not None:
        import yaml
        raw = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
        if not isinstance(raw, dict):
            raise ValueError(f'{file}: expected a mapping at the top level')
    cfg = _merged(DEFAULTS, raw)

    thresholds = {}
    for domain in THRESHOLD_DOMAINS:
        value = cfg['thresholds'].get(domain, DEFAULTS['thresholds'][domain])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f'thresholds.{domain} must be a number between 0 and 1, got {value!r}')
        thresholds[domain] = float(value)
    unknown = set(cfg['thresholds']) - set(THRESHOLD_DOMAINS)
    if unknown:
        raise ValueError(f'unknown threshold domain(s) {sorted(unknown)}; '
                         f'expected {list(THRESHOLD_DOMAINS)}')

    base = script_dir()
    paths, runtime, chunking = cfg['paths'], cfg['runtime'], cfg['chunking']
    return Settings(
        models_dir=_resolve_dir(paths.get('models'), base, 'models'),
        log_dir=_resolve_dir(paths.get('log'), base, 'log'),
        outputs_dir=_resolve_dir(paths.get('outputs'), base, 'outputs'),
        models=tuple(ModelSpec(key, m['repo'], m['folder'], m.get('revision'))
                     for key, m in cfg['models'].items()),
        device=str(runtime.get('device') or 'cpu'),
        batch_size=max(1, int(runtime.get('batch_size') or 16)),
        num_threads=int(runtime['num_threads']) if runtime.get('num_threads') else None,
        max_chunk_chars=int(chunking['max_chunk_chars']),
        overlap_units=int(chunking['overlap_units']),
        token_margin=int(chunking['token_margin']),
        thresholds=thresholds,
        source=file,
    )
