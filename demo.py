"""
FUSION TAPE demo — redact synthetic PII from every domain, with logging.

    python demo.py

The first run downloads the two models (~400 MB) into ./models; later runs load them
from disk.  Results are written to the directories set in settings.yaml (by default,
next to this script, all git-ignored):

    outputs/demo_redacted.txt   each input followed by its redacted text
    log/demo_log.json           the log returned by redact(..., log=True)
    log/fusion_tape.log         the operational run log (never contains input text)

Every name, number, address and credential below is made up.
"""
import json
import time

from fusion_tape import get_settings, redact

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


def main() -> None:
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


if __name__ == '__main__':
    main()
