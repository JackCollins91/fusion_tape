"""
FUSION TAPE demo — an example run with logging, then the benchmark regression run.

    python demo.py                    # example + benchmarks
    python demo.py --skip-benchmark   # example only

1. EXAMPLE.  Synthetic PII from every domain is redacted with ``log=True``:

       outputs/demo_redacted.txt   each input followed by its redacted text
       log/demo_log.json           the log returned by redact(..., log=True)
       log/fusion_tape.log         the operational run log (never contains input text)

2. BENCHMARKS.  The texts the pipeline was originally evaluated on (pii-detector
   evaluation notebook, §19: ai4privacy, Gretel and the long-document stress benchmark)
   are pulled into the git-ignored ``benchmarks/`` folder, redacted WITHOUT logging,
   and scored from the redacted text alone.  The result is written to
   ``.Benchmark-Report.md`` next to this script — commit it, and a later change that
   makes the pipeline leak more shows up as a diff.

The first run downloads the two models (~400 MB) into ./models.  Every name, number,
address and credential in the example below is made up.
"""
import argparse
import json
import os
import re
import shutil
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
BENCHMARK_FILE = 'pii_validation_sets.jsonl'              # one JSON object per text: dataset, id, text, gt
REPORT_FILE = HERE / '.Benchmark-Report.md'               # NOT git-ignored

# Where the benchmark texts come from, tried in order until one works:
#   1. benchmarks/pii_validation_sets.jsonl      already pulled in by an earlier run
#   2. $FUSION_TAPE_BENCHMARKS                    a file path or URL of your choice
#   3. ../pii-detector/pii_validation_sets.jsonl  a pii-detector checkout beside this repo
#   4. the pii-detector repository on GitHub      (private: set GITHUB_TOKEN to use it)
BENCHMARK_ENV = 'FUSION_TAPE_BENCHMARKS'
BENCHMARK_SIBLING = HERE.parent / 'pii-detector' / BENCHMARK_FILE
BENCHMARK_URL = ('https://raw.githubusercontent.com/JackCollins91/pii-detector/main/' + BENCHMARK_FILE)

# A PII value is scored only if it has at least this many letters/digits, and it has
# leaked when at least this many of them are still visible in the redacted text.
LEAK_MIN_CHARS = 3
BATCH_TEXTS = 32                                          # texts per redact() call

# Benchmark annotation label → FUSION TAPE domain.  UPPERCASE labels are ai4privacy's
# (also used by the long-document benchmark); snake_case labels are Gretel's.  Any
# other annotated label (dates, cities, job titles, …) is outside the nine domains
# and is not scored.
_ID_LABELS = [
    'ACCOUNTNUMBER', 'CREDITCARDNUMBER', 'CREDITCARDCVV', 'SSN', 'IBAN', 'BIC', 'VEHICLEVIN',
    'VEHICLEVRM', 'BITCOINADDRESS', 'ETHEREUMADDRESS', 'LITECOINADDRESS', 'MASKEDNUMBER',
    'PHONEIMEI', 'IP', 'IPV4', 'IPV6', 'MAC',
    'account_number', 'credit_card_number', 'ssn', 'customer_id', 'employee_id',
    'medical_record_number', 'bank_routing_number', 'device_identifier', 'unique_identifier',
    'biometric_identifier', 'certificate_license_number', 'license_plate', 'vehicle_identifier',
    'national_id', 'tax_id', 'health_plan_beneficiary_number', 'api_key', 'swift_bic', 'ipv4',
    'ipv6', 'cvv',
]
DOMAIN_LABELS = {
    'filepath': ['FILEPATH'],
    'username': ['USERNAME', 'user_name'],
    'id': _ID_LABELS,
    'organisation': ['COMPANYNAME', 'company_name'],
    'name': ['FIRSTNAME', 'LASTNAME', 'MIDDLENAME', 'PREFIX', 'first_name', 'last_name', 'name'],
    'telephone': ['PHONENUMBER', 'phone_number'],
    'address': ['STREET', 'STREETADDRESS', 'BUILDINGNUMBER', 'SECONDARYADDRESS', 'street_address',
                'address'],
    'password': ['PASSWORD', 'password'],
    'email': ['EMAIL', 'email'],
}
LABEL_DOMAIN = {label: domain for domain, labels in DOMAIN_LABELS.items() for label in labels}

# What the original evaluation measured on these same texts (pii-detector evaluation
# notebook §19; pii_pipeline_scores.json).  The report shows these beside each new run.
BASELINE = {
    'ai4privacy (selection)': {'pii': 1697, 'leaked': 11},
    'Gretel': {'pii': 3416, 'leaked': 4},
    'Benchmark (long docs)': {'pii': 8336, 'leaked': 40},
}
BASELINE_LEAKED_BY_DOMAIN = {'name': 20, 'username': 13, 'organisation': 13, 'address': 4,
                             'id': 4, 'password': 1, 'telephone': 0, 'email': 0, 'filepath': 0}

_MARKER_RE = re.compile(r'\[\[(?:%(t)s)(?:/(?:%(t)s))*\]\]' % {'t': '|'.join(DOMAIN_TAGS.values())})


def fetch_benchmarks() -> Path:
    """Make sure the benchmark texts are in the git-ignored benchmarks/ folder."""
    target = BENCHMARK_DIR / BENCHMARK_FILE
    if target.is_file():
        print(f'Benchmark texts: using {target}')
        return target
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    sources = [s for s in (os.environ.get(BENCHMARK_ENV), str(BENCHMARK_SIBLING), BENCHMARK_URL) if s]
    problems = []
    for source in sources:
        try:
            if source.startswith(('http://', 'https://')):
                request = urllib.request.Request(source)
                token = os.environ.get('GITHUB_TOKEN')
                if token and 'github' in source:
                    request.add_header('Authorization', f'Bearer {token}')
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                target.write_bytes(data)
            else:
                shutil.copyfile(Path(source).expanduser(), target)
            print(f'Benchmark texts: pulled from {source} → {target}')
            return target
        except Exception as exc:                          # try the next source
            problems.append(f'  {source}: {exc}')
    raise SystemExit(
        'Could not get the benchmark texts. Tried:\n' + '\n'.join(problems) +
        f'\nPut {BENCHMARK_FILE} in {BENCHMARK_DIR}, or set {BENCHMARK_ENV} to its path or URL '
        '(or GITHUB_TOKEN, to read it from the private pii-detector repository).')


def load_benchmarks(path: Path) -> dict[str, list[dict]]:
    """{benchmark name: [{'text', 'gt': [{'label', 'start', 'end', 'value'}]}]} in file order."""
    sets: dict[str, list[dict]] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sets.setdefault(row['dataset'], []).append(row)
    return sets


def visible_mask(original: str, redacted: str) -> bytearray:
    """
    Which characters of ``original`` are still readable in ``redacted``: 1 = visible,
    0 = taped over.  Uses the two strings only.  The text between markers is original
    text left untouched, so each such piece is located in ``original``; whatever lies
    between two located pieces is what a marker replaced.

    Pieces are located from the end of the text backwards, each at its last possible
    position.  A short piece such as ``","ip":"`` can occur more than once, and the
    output alone cannot say which occurrence it is; taking the last one matched the
    true censored spans on every text it was checked against.
    """
    pieces = _MARKER_RE.split(redacted)
    if len(pieces) == 1:                                  # no markers: nothing was censored
        if redacted != original:
            raise ValueError('redacted text differs from the original outside the markers')
        return bytearray(b'\x01' * len(original))
    mask = bytearray(len(original))
    end = len(original)
    for k in range(len(pieces) - 1, -1, -1):
        piece = pieces[k]
        if k == len(pieces) - 1:                          # the last piece is the end of the text
            at = end - len(piece) if original.endswith(piece) else -1
        elif k == 0:                                      # the first piece is its start
            at = 0 if original.startswith(piece) and len(piece) < end else -1
        else:                                             # a marker replaced at least one character
            at = original.rfind(piece, 0, end - 1) if piece else end
        if at < 0:
            raise ValueError('redacted text differs from the original outside the markers')
        mask[at:at + len(piece)] = b'\x01' * len(piece)
        end = at
    return mask


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
    lines.append(f'| **All** | {sum(r["texts"] for r in results.values()):,} | {words:,} | {total:,} | '
                 f'{leaked:,} | {pct(leaked, total)} | {base_leaked:,} of {base_total:,} '
                 f'({pct(base_leaked, base_total)}) |')

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
        '* The texts are the ones scored in the pii-detector evaluation notebook (§19): 1,000',
        '  from ai4privacy, 1,000 from Gretel, and the 60 long documents of the stress benchmark.',
        '* Each text is passed to `redact()` without logging. Scoring uses only the redacted',
        '  text: it is lined up with the original to see which characters are still readable.',
        '* A PII value is an annotation of one of the nine domains with at least '
        f'{LEAK_MIN_CHARS} letters or digits.',
        f'* It has **leaked** when {LEAK_MIN_CHARS} or more of its letters or digits are still readable.',
        '* The marker type is ignored: a name covered by `[[USERNAME]]` is not a leak.',
        '',
        '## Run',
        '',
        f'* Date: {date.today().isoformat()}',
        f'* Thresholds: {thresholds}',
        f'* Redaction time: {seconds / 60:.1f} minutes for {words:,} words '
        f'({words / max(seconds, 1e-9):.0f} words per second, batches of {BATCH_TEXTS} texts, '
        f'{settings.device})',
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
