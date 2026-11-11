"""
FUSION TAPE demo — an example run with logging, then the benchmark regression run.

    python demo.py                    # example + benchmarks
    python demo.py --skip-benchmark   # example only

1. EXAMPLE.  Synthetic PII from every domain is redacted with ``log=True``:

       outputs/demo_redacted.txt   each input followed by its redacted text
       log/demo_log.json           the log returned by redact(..., log=True)
       log/fusion_tape.log         the operational run log (never contains input text)

2. BENCHMARK.  The public Gretel PII dataset is downloaded from the Hugging Face Hub
   into the git-ignored ``benchmarks/`` folder (no account or token needed).  The
   1,000 of its texts that the pipeline was originally evaluated on are redacted
   WITHOUT logging and scored from the redacted text alone.  The result is written
   to ``.Benchmark-Report.md`` next to this script — commit it, and a later change
   that makes the pipeline leak more shows up as a diff.

The first run downloads the two models (~400 MB) into ./models.  Every name, number,
address and credential in the example below is made up.
"""
import argparse
import ast
import json
import random
import re
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

from fusion_tape import DOMAIN_TAGS, get_settings, redact

HERE = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
# 1. Example
# ══════════════════════════════════════════════════════════════════════════════
TEXTS = [
    # Names, an organisation, a telephone number and an e-mail address
    'Hi team, Dr. Eleanor Whitfield from Brightwater Analytics Ltd called this morning. '
    'You can reach her on +44 20 7946 0958 or at eleanor.whitfield@brightwater-analytics.co.uk.',

    # A username that hides a name, a password and a file path
    'Please reset the VPN account for marcus_oyelaran. His old password was Xk9#mLpQ2! and '
    r'the access report is saved in C:\Users\moyelaran\Documents\vpn_access_q3.xlsx.',

    # An address and identifiers (SSN, IBAN, customer number)
    'Ship the replacement card to 742 Evergreen Terrace, Apt 4B. The customer, Priya Raman, '
    'gave SSN 536-90-4399, IBAN GB29 NWBK 6016 1331 9268 19 and customer number CID-410603.',

    # Log-style text: handle, IP address, secret key, POSIX path
    'login ok user=@tess_marlowe ip=203.0.113.47 api_key=sk_live_9aF3kQ7zLm2Xw8Rt '
    'config=/home/tmarlowe/.config/fusion/credentials.yaml',

    # Nothing to redact
    'The quarterly review is on Thursday at ten; bring the printed agenda.',
]


def run_example() -> None:
    started = time.perf_counter()
    redacted, log = redact(TEXTS, log=True)
    elapsed = time.perf_counter() - started
    settings = get_settings()

    output_file = settings.outputs_dir / 'demo_redacted.txt'
    with output_file.open('w', encoding='utf-8') as f:
        for i, (original, censored) in enumerate(zip(TEXTS, redacted), 1):
            f.write(f'── Text {i} ─────────────────────────────\nIN : {original}\nOUT: {censored}\n\n')

    log_file = settings.log_dir / 'demo_log.json'
    log_file.write_text(json.dumps(
        [{'input': text, 'redacted': found, 'milliseconds': ms} for text, found, ms in log],
        indent=2, ensure_ascii=False), encoding='utf-8')

    for i, (censored, (_, found, ms)) in enumerate(zip(redacted, log), 1):
        print(f'\nText {i}  ({ms:.0f} ms, {len(found)} redaction(s))\n  {censored}')
        for value, marker in found.items():
            print(f'    {marker:<24} ← {value}')
    print(f'\n{len(TEXTS)} texts in {elapsed:.1f}s (including model loading)')
    print(f'Redacted text → {output_file}\nLog           → {log_file}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
BENCHMARK_DIR = HERE / 'benchmarks'                       # git-ignored
BENCHMARK_FILE = 'benchmark_texts.jsonl'                  # one JSON object per text: dataset, id, text, gt
REPORT_FILE = HERE / '.Benchmark-Report.md'               # NOT git-ignored

# The dataset is public on the Hugging Face Hub (Apache-2.0) and is downloaded without
# an account or token.  Nothing from it is committed: the downloads and the prepared
# texts stay in benchmarks/, which is git-ignored.
#   gretelai/gretel-pii-masking-en-v1   test + validation splits, 10,000 texts
HF = 'https://huggingface.co/datasets/'
GRETEL_URLS = [HF + 'gretelai/gretel-pii-masking-en-v1/resolve/main/data/test-00000-of-00001.parquet',
               HF + 'gretelai/gretel-pii-masking-en-v1/resolve/main/data/validation-00000-of-00001.parquet']
SAMPLE_TEXTS = 1_000      # texts scored
SAMPLE_SEED = 0
# The original evaluation used one seeded generator and drew 1,000 of 8,701 items from it
# before sampling Gretel.  That draw is repeated (and discarded) so the Gretel sample is the
# very one that was evaluated, and the baseline below stays comparable.
SAMPLE_SKIP = (8_701, 1_000)

# A PII value is scored only if it has at least this many letters/digits, and it has
# leaked when at least this many of them are still visible in the redacted text.
LEAK_MIN_CHARS = 3
BATCH_TEXTS = 32                                          # texts per redact() call

# Gretel annotation label → FUSION TAPE domain.  Any other annotated label (dates,
# cities, …) is outside the nine domains and is not scored.  Gretel does not annotate
# file paths, so that domain is not measured.
DOMAIN_LABELS = {
    'username': ['user_name'],
    'id': ['account_number', 'credit_card_number', 'ssn', 'customer_id', 'employee_id',
           'medical_record_number', 'bank_routing_number', 'device_identifier', 'unique_identifier',
           'biometric_identifier', 'certificate_license_number', 'license_plate',
           'vehicle_identifier', 'national_id', 'tax_id', 'health_plan_beneficiary_number',
           'api_key', 'swift_bic', 'ipv4', 'ipv6', 'cvv'],
    'organisation': ['company_name'],
    'name': ['first_name', 'last_name', 'name'],
    'telephone': ['phone_number'],
    'address': ['street_address', 'address'],
    'password': ['password'],
    'email': ['email'],
}
LABEL_DOMAIN = {label: domain for domain, labels in DOMAIN_LABELS.items() for label in labels}

# What the original evaluation measured on these same texts (pii-detector evaluation
# notebook, §19).  The report shows these beside each new run.
BASELINE = {'Gretel': {'pii': 3416, 'leaked': 4}}
BASELINE_LEAKED_BY_DOMAIN = {'username': 2, 'address': 1, 'id': 1}

_MARKER_RE = re.compile(r'\[\[(?:%(t)s)(?:/(?:%(t)s))*\]\]' % {'t': '|'.join(DOMAIN_TAGS.values())})


def download(url: str, folder: Path) -> Path:
    """Download ``url`` into ``folder`` (anonymous request) unless it is already there."""
    target = folder / url.rsplit('/', 1)[1]
    if not target.is_file():
        print(f'  downloading {url}')
        partial = target.with_name(target.name + '.partial')
        request = urllib.request.Request(url, headers={'User-Agent': 'fusion-tape-demo'})
        with urllib.request.urlopen(request, timeout=300) as response, partial.open('wb') as out:
            while block := response.read(1 << 20):
                out.write(block)
        partial.rename(target)
    return target


def gretel_rows(paths: list[Path]) -> list[dict]:
    """Gretel lists each PII value and its types but no offsets; find every occurrence."""
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise SystemExit('The Gretel benchmark is stored as parquet: pip install pyarrow') from None

    def entities(raw) -> list:
        if isinstance(raw, list):
            return raw
        for parse in (json.loads, ast.literal_eval):          # stored as a Python-literal string
            try:
                found = parse(raw)
                return found if isinstance(found, list) else []
            except Exception:
                continue
        return []

    def occurrences(text: str, value: str) -> list[tuple[int, int]]:
        def scan(haystack, needle):
            found, i = [], haystack.find(needle)
            while i >= 0:
                found.append((i, i + len(needle)))
                i = haystack.find(needle, i + 1)
            return found
        spans = scan(text, value)
        if not spans and len(text.lower()) == len(text) and len(value.lower()) == len(value):
            spans = scan(text.lower(), value.lower())
        return spans if len(value) >= 4 else spans[:1]         # a short value: first occurrence only

    rows = []
    for path in paths:
        for r in parquet.read_table(path).to_pylist():
            text, gt = r['text'], []
            for e in entities(r['entities']):
                value = str(e.get('entity', '')).strip()
                types = e.get('types') or []
                types = [types] if isinstance(types, str) else types
                if not value or not types:
                    continue
                for label in types:
                    for start, end in occurrences(text, value):
                        gt.append({'label': label, 'start': start, 'end': end, 'value': text[start:end]})
            rows.append({'id': str(r['uid']), 'text': text, 'gt': gt})
    return rows


def fetch_benchmarks() -> Path:
    """Make sure the benchmark texts are in the git-ignored benchmarks/ folder."""
    target = BENCHMARK_DIR / BENCHMARK_FILE
    if target.is_file():
        print(f'Benchmark texts: using {target}')
        return target
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    print('Benchmark texts: downloading the public dataset from the Hugging Face Hub')
    gretel = gretel_rows([download(url, BENCHMARK_DIR) for url in GRETEL_URLS])
    rng = random.Random(SAMPLE_SEED)
    rng.sample(range(SAMPLE_SKIP[0]), SAMPLE_SKIP[1])          # see SAMPLE_SKIP
    sets = {'Gretel': rng.sample(gretel, SAMPLE_TEXTS)}
    with target.open('w', encoding='utf-8') as f:
        for name, rows in sets.items():
            for row in rows:
                f.write(json.dumps({'dataset': name, **row}, ensure_ascii=False) + '\n')
    print(f'Benchmark texts: {sum(map(len, sets.values())):,} texts → {target}')
    return target


def load_benchmarks(path: Path) -> dict[str, list[dict]]:
    """{benchmark name: [{'text', 'gt': [{'label', 'start', 'end', 'value'}]}]} in file order."""
    sets: dict[str, list[dict]] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sets.setdefault(row['dataset'], []).append(row)
    return sets


def _locate(pieces: list[str], original: str, latest: bool) -> bytearray:
    """Place every piece in ``original``, each as early (or as late) as the others allow."""
    mask = bytearray(len(original))
    last = len(pieces) - 1
    edge = len(original) if latest else 0                 # where the previously placed piece begins/ends
    for k in (range(last, -1, -1) if latest else range(len(pieces))):
        piece = pieces[k]
        if k == 0:                                        # the first piece is the start of the text
            at = 0 if original.startswith(piece) else -1
        elif k == last:                                   # the last piece is its end
            at = len(original) - len(piece) if original.endswith(piece) else -1
        elif not piece:                                   # two markers side by side
            at = edge
        elif latest:                                      # a marker replaced at least one character
            at = original.rfind(piece, 0, edge - 1)
        else:
            at = original.find(piece, edge + 1)
        if at < 0:
            raise ValueError('redacted text differs from the original outside the markers')
        mask[at:at + len(piece)] = b'\x01' * len(piece)
        edge = at if latest else at + len(piece)
    return mask


def visible_mask(original: str, redacted: str) -> bytearray:
    """
    Which characters of ``original`` are still readable in ``redacted``: 1 = visible,
    0 = taped over.  Uses the two strings only.  The text between markers is original
    text left untouched, so each such piece is located in ``original``; whatever lies
    between two located pieces is what a marker replaced.

    A short piece can fit in more than one place: in ``[[ORGANISATION]] and
    [[ORGANISATION]]`` the visible " and " may be any " and " of "Will and Sons and
    Ziemann, Legros and Bogan", and the output alone cannot say which.  So the pieces
    are placed twice, as early and as late as possible, and a character counts as
    visible only when both placements agree.  On the benchmark texts this gives the
    same leaks as the pipeline's true censored spans.
    """
    pieces = _MARKER_RE.split(redacted)
    if len(pieces) == 1:                                  # no markers: nothing was censored
        if redacted != original:
            raise ValueError('redacted text differs from the original outside the markers')
        return bytearray(b'\x01' * len(original))
    earliest, latest = _locate(pieces, original, False), _locate(pieces, original, True)
    return bytearray(a & b for a, b in zip(earliest, latest))


def score_text(text: str, gt: list[dict], redacted: str) -> list[dict]:
    """
    One entry per scored PII value: {'domain', 'label', 'value', 'leaked'}.

    The same rules as the original evaluation: only annotations of the nine domains
    count; only letters and digits are counted; a value with fewer than LEAK_MIN_CHARS
    of them is ignored; a value has leaked when LEAK_MIN_CHARS or more of them are
    still visible.  WHICH marker covers a value does not matter — a name taped over
    as [[USERNAME]] is not a leak.
    """
    mask = visible_mask(text, redacted)
    spans: dict[tuple[int, int], str] = {}
    for e in gt:
        if e.get('label') in LABEL_DOMAIN and (e.get('end') or 0) > (e.get('start') or 0):
            spans.setdefault((e['start'], e['end']), e['label'])
    scored = []
    for (start, end), label in sorted(spans.items()):
        chars = [i for i in range(start, end) if text[i].isalnum()]
        if len(chars) < LEAK_MIN_CHARS:
            continue
        scored.append({'domain': LABEL_DOMAIN[label], 'label': label, 'value': text[start:end],
                       'leaked': sum(mask[i] for i in chars) >= LEAK_MIN_CHARS})
    return scored


def pct(part: int, whole: int) -> str:
    return f'{part / whole:.2%}' if whole else '—'


def run_benchmarks() -> None:
    sets = load_benchmarks(fetch_benchmarks())
    results = {}
    started = time.perf_counter()
    for name, rows in sets.items():
        texts = [row['text'] for row in rows]
        t0 = time.perf_counter()
        redacted: list[str] = []
        for b in range(0, len(texts), BATCH_TEXTS):
            out, log = redact(texts[b:b + BATCH_TEXTS])           # no logging
            assert log is None
            redacted += out
            print(f'\r  {name}: {len(redacted):,}/{len(texts):,} texts', end='', flush=True)
        seconds = time.perf_counter() - t0
        values = [v for row, red in zip(rows, redacted) for v in score_text(row['text'], row['gt'], red)]
        results[name] = {'texts': len(texts), 'words': sum(len(t.split()) for t in texts),
                         'seconds': seconds, 'values': values}
        leaked = sum(v['leaked'] for v in values)
        print(f'\r  {name}: {len(texts):,} texts, {leaked:,} of {len(values):,} PII values leaked '
              f'({pct(leaked, len(values))}) in {seconds:.0f}s')
    REPORT_FILE.write_text(build_report(results, time.perf_counter() - started), encoding='utf-8')
    print(f'Benchmark report → {REPORT_FILE}')


def build_report(results: dict, seconds: float) -> str:
    settings = get_settings()
    everything = [v for r in results.values() for v in r['values']]
    total, leaked = len(everything), sum(v['leaked'] for v in everything)
    base_total = sum(b['pii'] for b in BASELINE.values())
    base_leaked = sum(b['leaked'] for b in BASELINE.values())
    words = sum(r['words'] for r in results.values())

    lines = [
        '# FUSION TAPE — Benchmark Report',
        '',
        'Written by `python demo.py`. It is the regression test for the pipeline: the same',
        'benchmark texts are redacted on every run, so a change that makes FUSION TAPE leak',
        'more PII shows up here as a diff.',
        '',
        f'**Overall, {pct(leaked, total)} of PII leaked** — {leaked:,} of {total:,} PII values. '
        f'The original evaluation measured {pct(base_leaked, base_total)} '
        f'({base_leaked:,} of {base_total:,}) on the same texts.',
        '',
        '## By benchmark',
        '',
        '| Benchmark | Texts | Words | PII values | Leaked | % leaked | Original evaluation |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, r in results.items():
        n, k = len(r['values']), sum(v['leaked'] for v in r['values'])
        base = BASELINE.get(name)
        was = f"{base['leaked']:,} of {base['pii']:,} ({pct(base['leaked'], base['pii'])})" if base else '—'
        lines.append(f'| {name} | {r["texts"]:,} | {r["words"]:,} | {n:,} | {k:,} | {pct(k, n)} | {was} |')
    if len(results) > 1:
        lines.append(f'| **All** | {sum(r["texts"] for r in results.values()):,} | {words:,} | {total:,} | '
                     f'{leaked:,} | {pct(leaked, total)} | {base_leaked:,} of {base_total:,} '
                     f'({pct(base_leaked, base_total)}) |')

    lines += ['', '## Speed', '',
              f'Time spent inside `redact()`, in batches of {BATCH_TEXTS} texts on {settings.device}. '
              'Model loading and the dataset download are not included.', '',
              '| Benchmark | Words | Redaction time | Seconds per 1,000 words |',
              '|---|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f'| {name} | {r["words"]:,} | {r["seconds"]:.0f} s | '
                     f'{r["seconds"] / max(r["words"], 1) * 1000:.2f} |')

    lines += ['', '## By domain', '',
              'The domain is the one the benchmark annotated, not the marker that covered it.', '',
              '| Domain | PII values | Leaked | % leaked | Share of all leaks | Leaked in original evaluation |',
              '|---|---:|---:|---:|---:|---:|']
    n_by, k_by = Counter(v['domain'] for v in everything), Counter(v['domain'] for v in everything if v['leaked'])
    for domain in sorted(DOMAIN_TAGS, key=lambda d: (-k_by[d], -n_by[d])):
        lines.append(f'| {DOMAIN_TAGS[domain]} | {n_by[domain]:,} | {k_by[domain]:,} | '
                     f'{pct(k_by[domain], n_by[domain])} | {pct(k_by[domain], leaked)} | '
                     f'{BASELINE_LEAKED_BY_DOMAIN.get(domain, 0):,} |')

    lines += ['', '## Leaked values', '',
              'Every value below is synthetic benchmark data.', '']
    for name, r in results.items():
        found = Counter((v['domain'], v['label'], v['value']) for v in r['values'] if v['leaked'])
        lines += [f'**{name}** — {sum(found.values()):,} leaked', '']
        if not found:
            lines += ['None.', '']
            continue
        lines += ['| Domain | Benchmark label | Value | Times |', '|---|---|---|---:|']
        for (domain, label, value), times in sorted(found.items()):
            shown = ' '.join(value.split()).replace('|', '\\|').replace('`', "'")
            lines.append(f'| {DOMAIN_TAGS[domain]} | {label} | `{shown}` | {times} |')
        lines.append('')

    thresholds = ', '.join(f'{d} {t:g}' for d, t in settings.thresholds.items())
    lines += [
        '## How this is scored',
        '',
        '* The texts come from the public dataset `gretelai/gretel-pii-masking-en-v1` on the',
        f'  Hugging Face Hub (test and validation splits). {SAMPLE_TEXTS:,} texts are drawn with a fixed',
        '  seed, which gives the same texts the pipeline was originally evaluated on.',
        '* Each text is passed to `redact()` without logging. Scoring uses only the redacted',
        '  text: it is lined up with the original to see which characters are still readable.',
        '  Where a short stretch of visible text could sit in more than one place, a character',
        '  counts as readable only if it is readable wherever that stretch is placed.',
        '* A PII value is an annotation of one of the nine domains with at least '
        f'{LEAK_MIN_CHARS} letters or digits.',
        f'* It has **leaked** when {LEAK_MIN_CHARS} or more of its letters or digits are still readable.',
        '* The marker type is ignored: a name covered by `[[USERNAME]]` is not a leak.',
        '',
        '## Run',
        '',
        f'* Date: {date.today().isoformat()}',
        f'* Thresholds: {thresholds}',
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='FUSION TAPE demo and benchmark regression run.')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='run the example only (the benchmark run takes several minutes)')
    args = parser.parse_args()
    print('══ Example ══════════════════════════════════════════')
    run_example()
    if not args.skip_benchmark:
        print('\n══ Benchmarks ═══════════════════════════════════════')
        run_benchmarks()


if __name__ == '__main__':
    main()
