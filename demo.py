"""
FUSION TAPE demo.

    python demo.py                    # example + benchmark
    python demo.py --skip-benchmark   # example only

1. EXAMPLE    redacts a few synthetic texts with logging and writes
              outputs/demo_redacted.txt and log/demo_log.json.

2. BENCHMARK  downloads the public Gretel PII dataset into the git-ignored benchmarks/
              folder (no account or token needed), redacts 1,000 of its texts without
              logging, and writes .Benchmark-Report.md: how much PII leaked, in which
              domains, and how fast.  Commit the report; a later change that makes the
              pipeline leak more then shows up as a diff.

The first run also downloads the two models (~400 MB) into ./models.
"""
import ast
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

import importlib

HERE = Path(__file__).resolve().parent

# This script sits inside the package folder, so import that folder by its own name.
sys.path.insert(0, str(HERE.parent))
fusion_tape = importlib.import_module(HERE.name)
DOMAIN_TAGS, get_settings, redact = fusion_tape.DOMAIN_TAGS, fusion_tape.get_settings, fusion_tape.redact

# ── 1. Example ────────────────────────────────────────────────────────────────
# Every name, number, address and credential here is made up.
TEXTS = [
    'Hi team, Dr. Eleanor Whitfield from Brightwater Analytics Ltd called this morning. '
    'You can reach her on +44 20 7946 0958 or at eleanor.whitfield@brightwater-analytics.co.uk.',

    'Please reset the VPN account for marcus_oyelaran. His old password was Xk9#mLpQ2! and '
    r'the access report is saved in C:\Users\moyelaran\Documents\vpn_access_q3.xlsx.',

    'Ship the replacement card to 742 Evergreen Terrace, Apt 4B. The customer, Priya Raman, '
    'gave SSN 536-90-4399, IBAN GB29 NWBK 6016 1331 9268 19 and customer number CID-410603.',

    'login ok user=@tess_marlowe ip=203.0.113.47 api_key=sk_live_9aF3kQ7zLm2Xw8Rt '
    'config=/home/tmarlowe/.config/fusion/credentials.yaml',

    'The quarterly review is on Thursday at ten; bring the printed agenda.',
]


def run_example() -> None:
    redacted, log = redact(TEXTS, log=True)
    settings = get_settings()

    output_file = settings.outputs_dir / 'demo_redacted.txt'
    output_file.write_text(''.join(f'IN : {text}\nOUT: {censored}\n\n'
                                   for text, censored in zip(TEXTS, redacted)), encoding='utf-8')
    log_file = settings.log_dir / 'demo_log.json'
    log_file.write_text(json.dumps(
        [{'input': text, 'redacted': found, 'milliseconds': ms} for text, found, ms in log],
        indent=2, ensure_ascii=False), encoding='utf-8')

    for censored, (_, found, ms) in zip(redacted, log):
        print(f'\n{censored}\n  ({ms:.0f} ms)')
        for value, marker in found.items():
            print(f'  {marker:<24} ← {value}')
    print(f'\nRedacted text → {output_file}\nLog           → {log_file}')


# ── 2. Benchmark ──────────────────────────────────────────────────────────────
BENCHMARK_DIR = HERE / 'benchmarks'               # git-ignored
REPORT_FILE = HERE / '.Benchmark-Report.md'       # committed
GRETEL = 'https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1/resolve/main/data/'
GRETEL_FILES = ['test-00000-of-00001.parquet', 'validation-00000-of-00001.parquet']   # Apache-2.0
N_TEXTS, SEED = 1_000, 0                          # a fixed sample, so every run scores the same texts
LEAK_MIN_CHARS = 3                                # a value has leaked when this many letters/digits show

# Gretel label → FUSION TAPE domain.  Other labels (dates, cities, …) are not scored,
# and Gretel has no file paths.
LABEL_DOMAIN = {
    'first_name': 'name', 'last_name': 'name', 'name': 'name',
    'user_name': 'username',
    'email': 'email',
    'phone_number': 'telephone',
    'street_address': 'address', 'address': 'address',
    'company_name': 'organisation',
    'password': 'password',
    **dict.fromkeys([
        'account_number', 'credit_card_number', 'ssn', 'customer_id', 'employee_id',
        'medical_record_number', 'bank_routing_number', 'device_identifier', 'unique_identifier',
        'biometric_identifier', 'certificate_license_number', 'license_plate', 'vehicle_identifier',
        'national_id', 'tax_id', 'health_plan_beneficiary_number', 'api_key', 'swift_bic', 'ipv4',
        'ipv6', 'cvv'], 'id'),
}
MARKER = re.compile(r'\[\[[A-Z/]+\]\]')


def load_gretel() -> list[dict]:
    """Download Gretel (once) and return N_TEXTS texts as {'text', 'pii': [(start, end, domain)]}."""
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        sys.exit('The benchmark reads parquet files: pip install pyarrow')
    BENCHMARK_DIR.mkdir(exist_ok=True)
    rows = []
    for name in GRETEL_FILES:
        path = BENCHMARK_DIR / name
        if not path.is_file():
            print(f'downloading {GRETEL + name}')
            urllib.request.urlretrieve(GRETEL + name, path)
        rows += parquet.read_table(path).to_pylist()

    texts = []
    for row in random.Random(SEED).sample(rows, N_TEXTS):
        text, pii = row['text'], set()
        # Gretel gives each PII value and its types, but not where it is: find every occurrence.
        for entity in ast.literal_eval(row['entities']):
            value = str(entity['entity']).strip()
            domains = {LABEL_DOMAIN[t] for t in entity['types'] if t in LABEL_DOMAIN}
            for match in re.finditer(re.escape(value), text) if value and domains else ():
                pii.add((match.start(), match.end(), min(domains)))
        texts.append({'text': text, 'pii': sorted(pii)})
    return texts


def visible(original: str, redacted: str) -> bytearray:
    """
    1 for every character of ``original`` still readable in ``redacted``, else 0 — worked
    out from the two strings alone.  The pieces of text between markers are located in the
    original; what lies between two pieces is what a marker replaced.

    A short piece can fit in more than one place (the " and " of "[[X]] and [[Y]]" when the
    censored names contain " and " too), so the pieces are placed twice, as early and as
    late as possible, and a character counts as readable only where both agree.
    """
    pieces = MARKER.split(redacted)

    def place(latest: bool) -> bytearray:
        mask, last = bytearray(len(original)), len(pieces) - 1
        edge = len(original) if latest else 0
        for k in (range(last, -1, -1) if latest else range(last + 1)):
            piece = pieces[k]
            if k == 0:
                at = 0
            elif k == last:
                at = len(original) - len(piece)
            elif latest:
                at = original.rfind(piece, 0, edge - 1) if piece else edge
            else:
                at = original.find(piece, edge + 1) if piece else edge
            if at < 0 or original[at:at + len(piece)] != piece:
                raise ValueError('redacted text differs from the original outside the markers')
            mask[at:at + len(piece)] = b'\x01' * len(piece)
            edge = at if latest else at + len(piece)
        return mask

    return bytearray(a & b for a, b in zip(place(False), place(True)))


def score(text: str, pii: list[tuple], redacted: str) -> list[tuple[str, str, bool]]:
    """(domain, value, leaked) for every PII value with at least LEAK_MIN_CHARS letters/digits.
    Which marker covers a value does not matter: a name under [[USERNAME]] is not a leak."""
    mask, out = visible(text, redacted), []
    for start, end, domain in pii:
        chars = [i for i in range(start, end) if text[i].isalnum()]
        if len(chars) >= LEAK_MIN_CHARS:
            out.append((domain, text[start:end], sum(mask[i] for i in chars) >= LEAK_MIN_CHARS))
    return out


def run_benchmark() -> None:
    rows = load_gretel()
    texts = [row['text'] for row in rows]
    started = time.perf_counter()
    redacted = [r for i in range(0, len(texts), 32) for r in redact(texts[i:i + 32])[0]]   # no logging
    seconds = time.perf_counter() - started

    values = [v for row, red in zip(rows, redacted) for v in score(row['text'], row['pii'], red)]
    leaked = [v for v in values if v[2]]
    words = sum(len(t.split()) for t in texts)
    pct = lambda part, whole: f'{part / whole:.2%}' if whole else '—'
    total_by, leaked_by = Counter(v[0] for v in values), Counter(v[0] for v in leaked)

    report = [
        '# FUSION TAPE — Benchmark Report',
        '',
        'Written by `python demo.py`. The same texts are redacted on every run, so a change',
        'that makes FUSION TAPE leak more PII shows up here as a diff.',
        '',
        f'**{pct(len(leaked), len(values))} of PII leaked** — {len(leaked):,} of {len(values):,} PII values '
        f'in {len(texts):,} texts from `gretelai/gretel-pii-masking-en-v1`.',
        '',
        '## Leaks by domain',
        '',
        '| Domain | PII values | Leaked | % leaked |',
        '|---|---:|---:|---:|',
        *[f'| {DOMAIN_TAGS[d]} | {total_by[d]:,} | {leaked_by[d]:,} | {pct(leaked_by[d], total_by[d])} |'
          for d in sorted(DOMAIN_TAGS, key=lambda d: (-leaked_by[d], -total_by[d]))],
        '',
        '## Leaked values',
        '',
        *[f'* {DOMAIN_TAGS[domain]}: `{value}`' for domain, value, _ in sorted(leaked)],
        '',
        '## Speed',
        '',
        '| Words | Redaction time | Seconds per 1,000 words |',
        '|---:|---:|---:|',
        f'| {words:,} | {seconds:.0f} s | {seconds / words * 1000:.2f} |',
        '',
        '## How this is scored',
        '',
        f'* {N_TEXTS:,} texts are drawn with a fixed seed from the test and validation splits.',
        '* Each is passed to `redact()` without logging; only the redacted text is scored.',
        f'* A PII value is an annotation of one of the nine domains with at least {LEAK_MIN_CHARS} '
        'letters or digits.',
        f'* It has leaked when {LEAK_MIN_CHARS} or more of them are still readable. The marker type is '
        'ignored: a name covered by `[[USERNAME]]` is not a leak.',
        '* Gretel does not annotate file paths, so FILEPATH is not measured.',
        '',
        f'Run on {date.today().isoformat()} with thresholds '
        + ', '.join(f'{d} {t:g}' for d, t in get_settings().thresholds.items()) + '.',
        '',
    ]
    REPORT_FILE.write_text('\n'.join(report), encoding='utf-8')
    print(f'{len(leaked):,} of {len(values):,} PII values leaked ({pct(len(leaked), len(values))}); '
          f'{seconds / words * 1000:.2f} s per 1,000 words\nBenchmark report → {REPORT_FILE}')


if __name__ == '__main__':
    run_example()
    if '--skip-benchmark' not in sys.argv:
        print('\nRunning the benchmark (about a minute) …')
        run_benchmark()
