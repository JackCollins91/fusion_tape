"""
FUSION TAPE — an English-language PII redactor pipeline.

    from fusion_tape import redact

    redacted, log = redact('Call Jane Doe on +44 20 7946 0958')
    # redacted == 'Call [[NAME]] [[NAME]] on [[TELEPHONE]]',  log is None

    redacted, log = redact(['first text', 'second text'], log=True)
    # redacted is a list; log is [(input, {'pii string': '[[MARKER]]'}, milliseconds), ...]

    redacted, log = redact('Call Jane Doe on +44 20 7946 0958', skip=['names', 'usernames'])
    # the NAME and USERNAME pipes are left out

Everything is configured in ``settings.yaml``.  Models are downloaded on the first
call only and nothing but those downloads ever touches the network.
"""
from __future__ import annotations

from .spans import DOMAIN_TAGS, DOMAINS, SKIP_KEYWORDS, skipped_domains  # noqa: F401

__all__ = ['redact', 'get_settings', 'DOMAINS', 'DOMAIN_TAGS', 'SKIP_KEYWORDS']
__version__ = '0.2.1'


def redact(text: str | list[str], log: bool = False, skip: str | list[str] | None = None):
    """
    Replace PII in ``text`` with censorship markers such as ``[[EMAIL]]`` or
    ``[[NAME/USERNAME]]`` (every domain that flagged the string).

    Parameters
    ----------
    text : a string, or a list of strings (processed together in batches).
    log  : also return a per-text log.
    skip : a keyword, or a list of keywords, naming domain pipes to leave out of the
           pipeline.  The keywords are the words of the FUSION TAPE acronym:
           ``filepaths``, ``usernames``, ``ids``, ``organisations``, ``names``,
           ``telephone``, ``address``, ``password``, ``email`` (any letter case).
           Anything else raises ``ValueError``.

    Returns
    -------
    (redacted, log)
        redacted : ``str`` for a string input, ``list[str]`` for a list — the input
                   with nothing changed except PII replaced by markers.
        log      : ``None`` when ``log=False``; otherwise a list with one tuple per
                   input text: ``(input_text, {redacted_string: marker}, milliseconds)``.
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)
    if not all(isinstance(t, str) for t in texts):
        raise TypeError('text must be a string or a list of strings')
    skipped = skipped_domains(skip)                      # validated before any model is loaded
    from .engine import get_engine
    redacted, entries = get_engine().redact_many(texts, skipped)
    return (redacted[0] if single else redacted), (entries if log else None)


def get_settings():
    """The active settings (directories, thresholds, …) — loads the models if not loaded yet."""
    from .engine import get_engine
    return get_engine().settings
