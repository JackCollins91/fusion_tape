"""
Rule-based (regex) PII detectors, as benchmarked in the pii-detector evaluation notebook
(the §-numbers below refer to its sections).

Every detector has the signature ``detect(text) -> [{'word', 'entity_group', 'start', 'end'}]``
with offsets into ``text``.  They are pure Python and need no model.

  naive_password_detect   §5   symbol rule + machine-random-token rule + secret-named keys
  naive_street_detect     §6   [number] Capitalised Name(s) + street suffix, secondary units
  naive_email_detect      §7   RFC-style e-mail regex
  naive_name_detect       §8c  honorific + name, bare honorific, greeting context
  naive_org_detect        §9b  legal/institutional suffixes, "Name - Name", "A, B and C"
  naive_uid_detect        §10  SSN/NINO/IBAN/cards/IP/MAC/VIN/prefixed + segmented IDs + crypto wallets
  naive_username_detect   §11c keyword context, @handle, first.last, Name65, lower63
  naive_phone_detect      §12  worldwide candidate regex + structural validation
  naive_filepath_detect   §13  Windows/UNC/env/POSIX/home/shell/dot-relative/bare/file:// paths
"""
from __future__ import annotations

import re
import unicodedata

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════
def _is_currency(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch) == 'Sc'


def _prev_nonspace(text: str, i: int) -> str:
    j = i - 1
    while j >= 0 and text[j] in ' \t\u00A0\u202F':
        j -= 1
    return text[j] if j >= 0 else ''


_CAP_WORD = r"[A-Z][\w'’.\-]*"
_STOP_FIRST = {'The', 'A', 'An', 'This', 'That', 'Our', 'Your', 'My', 'Their', 'His', 'Her',
               'Its', 'At', 'On', 'In', 'To', 'From', 'Visit', 'Near', 'Via', 'And', 'Or'}


# ═════════════════════════════════════════════════════════════════════════════
# §5 Password
# ═════════════════════════════════════════════════════════════════════════════
_PASSWD_TOKEN_RE = re.compile(r'\S+')
_PWD_SYMBOL_RE = re.compile(r'[!@#$%^&*+=]')
_EXCL_URL = re.compile(r'https?://|ftp://', re.I)
_EXCL_EMAIL = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
_EXCL_DATE = re.compile(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$')
_STRIP_CHARS = '.,;:!?"\'`()[]{}*'      # '*': markdown bold around **$82.50**
_KEY_VALUE_RE = re.compile(r'^([A-Za-z_][\w.\-]{0,30})[=:]')
# A key that names a secret makes its value a password whatever the value looks like.
_SECRET_KEY_RE = re.compile(r'(?i)pass|pwd|(?:^|[_.\-])pw(?:$|[_.\-])|secret|token|api[_.\-]?key|auth|cred|(?:^|[_.\-])pin(?:$|[_.\-])')
_RANDOM_TOKEN_RE = re.compile(r'[A-Za-z0-9!@#$%^&*+=_~]{8,64}')


def _char_class(ch: str) -> str:
    return 'l' if ch.islower() else 'u' if ch.isupper() else 'd' if ch.isdigit() else 's'


def _looks_random(tok: str) -> bool:
    """Machine-generated secret: 8–64 chars, has letters, switches character class often."""
    if not _RANDOM_TOKEN_RE.fullmatch(tok) or not any(c.isalpha() for c in tok):
        return False
    classes = {_char_class(c) for c in tok}
    switches = sum(_char_class(a) != _char_class(b) for a, b in zip(tok, tok[1:]))
    upper_runs = re.findall(r'[A-Z]{2,}', tok)
    if len(classes) >= 3:
        return switches >= 4
    if classes == {'l', 'u'}:                       # letters only: need an UPPER run too
        return switches >= 4 and bool(upper_runs)
    return False


def naive_password_detect(text: str) -> list[dict]:
    """
    1. 8+ chars with a digit AND a password symbol (!@#$%^&*+=)
    2. machine-random tokens (mixed classes, frequent class switches)
    3. the value of a secret-named key (pw=…, api_key: …) whatever it looks like
    Excludes URLs, e-mails, dates and currency amounts.
    """
    results = []
    for m in _PASSWD_TOKEN_RE.finditer(text):
        raw = m.group()
        if _EXCL_URL.search(raw) or _EXCL_EMAIL.match(raw):
            continue
        token = raw.strip(_STRIP_CHARS)
        kv = _KEY_VALUE_RE.match(token)          # session=<uuid>, ua=Mozilla/5.0, pw=<secret>
        secret_key = bool(kv and _SECRET_KEY_RE.search(kv.group(1)))
        if kv:
            token = token[kv.end():].strip(_STRIP_CHARS)   # judge the value, not "key="
        if secret_key and len(token) >= 4:
            s = m.start() + raw.find(token)
            results.append({'word': token, 'entity_group': 'password', 'start': s, 'end': s + len(token)})
            continue
        if len(token) < 8 or _EXCL_DATE.match(token):
            continue
        if (_is_currency(token[0]) and re.fullmatch(r'.\s?[\d,.]+', token)) or \
           re.fullmatch(r'[\d,.]+', token):
            continue                                  # money / plain numbers
        rule1 = bool(re.search(r'\d', token) and _PWD_SYMBOL_RE.search(token))
        if rule1 or _looks_random(token):
            s = m.start() + raw.find(token)
            results.append({'word': token, 'entity_group': 'password',
                            'start': s, 'end': s + len(token)})
    return results


# ═════════════════════════════════════════════════════════════════════════════
# §6 Street address
# ═════════════════════════════════════════════════════════════════════════════
_STREET_SUFFIXES = sorted({
    'alley', 'avenue', 'bend', 'boulevard', 'branch', 'bridge', 'brook', 'brooks', 'burg',
    'bypass', 'camp', 'canyon', 'cape', 'causeway', 'center', 'centre', 'chase', 'circle',
    'circles', 'cliff', 'cliffs', 'close', 'club', 'common', 'commons', 'corner', 'corners',
    'course', 'court', 'courts', 'cove', 'coves', 'creek', 'crescent', 'crest', 'croft',
    'crossing', 'crossroad', 'curve', 'dale', 'dam', 'divide', 'drive', 'drives', 'estate',
    'estates', 'expressway', 'extension', 'falls', 'ferry', 'field', 'fields', 'flat',
    'flats', 'ford', 'fords', 'forest', 'forge', 'forges', 'fork', 'forks', 'fort',
    'freeway', 'garden', 'gardens', 'garth', 'gate', 'gateway', 'glen', 'glens', 'green',
    'greens', 'grove', 'groves', 'harbor', 'harbors', 'harbour', 'haven', 'heights',
    'highway', 'hill', 'hills', 'hollow', 'inlet', 'island', 'islands', 'isle', 'junction',
    'junctions', 'key', 'keys', 'knoll', 'knolls', 'lake', 'lakes', 'landing', 'lane',
    'light', 'lights', 'loaf', 'lock', 'locks', 'lodge', 'loop', 'mall', 'manor', 'manors',
    'meadow', 'meadows', 'mews', 'mill', 'mills', 'mission', 'motorway', 'mount',
    'mountain', 'mountains', 'neck', 'orchard', 'oval', 'overpass', 'parade', 'park',
    'parks', 'parkway', 'parkways', 'pass', 'passage', 'path', 'pike', 'pine', 'pines',
    'place', 'plain', 'plains', 'plaza', 'point', 'points', 'port', 'ports', 'prairie',
    'quay', 'radial', 'ramp', 'ranch', 'rapid', 'rapids', 'rest', 'ridge', 'ridges', 'rise',
    'river', 'road', 'roads', 'route', 'row', 'rue', 'run', 'shoal', 'shoals', 'shore',
    'shores', 'skyway', 'spring', 'springs', 'spur', 'spurs', 'square', 'squares',
    'station', 'stravenue', 'stream', 'street', 'streets', 'summit', 'terrace',
    'throughway', 'trace', 'track', 'trafficway', 'trail', 'tunnel', 'turnpike',
    'underpass', 'union', 'unions', 'vale', 'valley', 'valleys', 'viaduct', 'view',
    'views', 'village', 'villages', 'ville', 'vista', 'walk', 'walks', 'wall', 'way', 'ways',
    'well', 'wells', 'wharf', 'yard',
}, key=len, reverse=True)
_STREET_ABBREVS = ['St', 'Ave', 'Av', 'Blvd', 'Rd', 'Dr', 'Ln', 'Ct', 'Pl', 'Cres', 'Ter',
                   'Tce', 'Hwy', 'Pkwy', 'Sq', 'Cir', 'Trl', 'Ter', 'Way', 'Pde']
_STREET_RE = re.compile(
    r'(?:(?<![\w.])\d{1,6}[A-Za-z]?(?:-\d{1,6})?,?\s+)?'            # building number
    rf'(?:{_CAP_WORD}\s+){{1,4}}'                                    # street name words
    r'(?:(?:' + '|'.join(w.title() for w in _STREET_SUFFIXES) + '|'
    + '|'.join(w.upper() for w in _STREET_SUFFIXES) + r')\b'           # Club / CLUB (not 'club')
    r'|(?:' + '|'.join(_STREET_ABBREVS) + r')\b\.?)'                 # or abbreviation
)
_SECONDARY_RE = re.compile(
    r'\b(?:Apt|Apartment|Suite|Ste|Unit|Flat|Floor|Fl|Room|Rm|Bldg|Building)\.?\s*#?\s*'
    r'\d{1,5}[A-Za-z]?\b|#\s?\d{1,5}[A-Za-z]?\b')


def naive_street_detect(text: str) -> list[dict]:
    """Street-address spans: [number] Capitalised Name(s) + suffix, and secondary units."""
    out = []
    for m in _STREET_RE.finditer(text):
        s, e = m.start(), m.end()
        words = text[s:e].split()
        while words and words[0] in _STOP_FIRST:              # drop a leading stop-word
            s = text.index(words[1], s) if len(words) > 1 else e
            words = words[1:]
        if len(words) >= 2 and e > s:
            out.append({'word': text[s:e], 'entity_group': 'address', 'start': s, 'end': e})
    for m in _SECONDARY_RE.finditer(text):
        out.append({'word': m.group(), 'entity_group': 'address',
                    'start': m.start(), 'end': m.end()})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §7 E-mail
# ═════════════════════════════════════════════════════════════════════════════
_EMAIL_DETECT_RE = re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b')


def naive_email_detect(text: str) -> list[dict]:
    return [{'word': m.group(), 'entity_group': 'email', 'start': m.start(), 'end': m.end()}
            for m in _EMAIL_DETECT_RE.finditer(text)]


# ═════════════════════════════════════════════════════════════════════════════
# §8c Name rules
# ═════════════════════════════════════════════════════════════════════════════
_HONORIFICS = r'(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Dame|Lord|Lady|Rev|Fr|Capt|Col|Gen|Lt|Sgt|Hon|Mme|Mlle|Herr|Frau)'
_GREETING_SKIP = {'Team', 'All', 'Everyone', 'Sir', 'Madam', 'There', 'Friends', 'Colleagues',
                  'Folks', 'Guys', 'Customer', 'Client', 'Patient', 'Manager', 'Support', 'Hiring',
                  'Participant', 'Participants', 'Applicant', 'Candidate', 'Member', 'Members',
                  'Valued', 'User', 'Users', 'Students', 'Staff', 'Parents', 'Internal', 'Damaged'}
_NAME_RES = [
    # Honorific + up to three capitalised words (also "Mrs.Auer" with no space)
    re.compile(rf"\b{_HONORIFICS}\.?\s*[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){{0,2}}"),
    # A bare honorific used as a stand-in for a name: "Hello Ms.," "Sincerely, Mr."
    re.compile(rf"\b{_HONORIFICS}\.(?=\s*(?:[,;:!?]|$))"),
]
_GREETING_RE = re.compile(r"\b(?:Dear|Hi|Hello|Hey|Greetings|Thanks|Thank you),?\s+([A-Z][a-z'’\-]+)(?:\s+([A-Z][a-z'’\-]+))?")


def naive_name_detect(text: str) -> list[dict]:
    out = []
    for pat in _NAME_RES:
        for m in pat.finditer(text):
            out.append({'word': m.group(), 'entity_group': 'name', 'start': m.start(), 'end': m.end()})
    for m in _GREETING_RE.finditer(text):
        if m.group(1) in _GREETING_SKIP:
            continue
        s = m.start(1)
        e = m.end(2) if m.group(2) and m.group(2) not in _GREETING_SKIP else m.end(1)
        out.append({'word': text[s:e], 'entity_group': 'name', 'start': s, 'end': e})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §9b Organisation rules
# ═════════════════════════════════════════════════════════════════════════════
_ORG_SUFFIX = (r'(?:Inc|Incorporated|Ltd|Limited|LLC|L\.L\.C|LLP|PLC|plc|Corp|Corporation|Co|'
               r'Company|Group|Holdings|GmbH|AG|SA|S\.A|SAS|BV|NV|Pty|Partners|Associates|'
               r'Bank|Foundation|Trust|Industries|Enterprises|Solutions|Technologies|Systems|'
               r'Consulting|Labs|Ventures|Capital|Hospital|University|College|Institute|Clinic)')
_ORG_RES = [
    re.compile(rf"(?:{_CAP_WORD}[,&]?\s+){{1,4}}(?:&\s+)?{_ORG_SUFFIX}\b\.?"),                 # Acme Holdings Ltd
    re.compile(r"\b[A-Z][a-z]+(?:\s-\s|-)[A-Z][a-z]+(?:,?\s+(?:Inc|LLC|Ltd|PLC|Group)\b\.?)?"),  # Franey - Boyle, Mcbride-Jones
    re.compile(r"\b[A-Z][a-z]+, [A-Z][a-z]+,? and [A-Z][a-z]+\b"),                           # Stracke, Rohan and Steuber
    re.compile(r"\b[A-Z][a-z]+ (?:and|&) Sons\b"),
]


def naive_org_detect(text: str) -> list[dict]:
    out = []
    for pat in _ORG_RES:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            words = text[s:e].split()
            while words and words[0] in _STOP_FIRST:
                s = text.index(words[1], s) if len(words) > 1 else e
                words = words[1:]
            if e - s >= 4:
                out.append({'word': text[s:e], 'entity_group': 'organisation', 'start': s, 'end': e})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §10 Unique identifiers (incl. crypto wallets)
# ═════════════════════════════════════════════════════════════════════════════
_B58 = r'[1-9A-HJ-NP-Za-km-z]'
_CRYPTO_RES = [
    ('bitcoin',  re.compile(rf'(?<![\w])[13]{_B58}{{25,40}}(?![\w])')),          # legacy / P2SH
    ('bitcoin',  re.compile(r'(?<![\w])(?:bc1|tb1)[02-9ac-hj-np-z]{11,71}(?![\w])', re.I)),
    ('ethereum', re.compile(r'(?<![\w])0x[0-9a-fA-F]{40}(?![\w])')),
    ('litecoin', re.compile(rf'(?<![\w])[LM]{_B58}{{25,40}}(?![\w])')),
    ('litecoin', re.compile(r'(?<![\w])ltc1[02-9ac-hj-np-z]{11,71}(?![\w])', re.I)),
    ('monero',   re.compile(rf'(?<![\w])[48][0-9AB]{_B58}{{93}}(?![\w])')),
    ('tron',     re.compile(rf'(?<![\w])T{_B58}{{33}}(?![\w])')),
    ('ripple',   re.compile(rf'(?<![\w])r{_B58}{{24,34}}(?![\w])')),
    ('dogecoin', re.compile(rf'(?<![\w])D[5-9A-HJ-NP-U]{_B58}{{32}}(?![\w])')),
    ('solana',   re.compile(rf'(?<![\w]){_B58}{{32,44}}(?![\w])')),
]


def naive_crypto_detect(text: str) -> list[dict]:
    """Crypto wallet addresses; base58 candidates must mix upper, lower and digits."""
    out, seen = [], set()
    for coin, pat in _CRYPTO_RES:
        for m in pat.finditer(text):
            v = m.group()
            if (m.start(), m.end()) in seen:
                continue
            if not v.lower().startswith(('bc1', 'tb1', 'ltc1', '0x')):
                if not (re.search(r'[a-z]', v) and re.search(r'[A-Z]', v) and re.search(r'\d', v)):
                    continue
            seen.add((m.start(), m.end()))
            out.append({'word': v, 'entity_group': f'crypto:{coin}',
                        'start': m.start(), 'end': m.end()})
    return out


_UID_RES = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                                     # US SSN
    re.compile(r'\b756[.\s]?\d{4}[.\s]?\d{4}(?:[.\s]?\d{2})?\b'),            # Swiss AHV
    re.compile(r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b'),             # UK NINO (ZZ 998152 T)
    re.compile(r'\b\d{3}[ -]\d{3}[ -]\d{3}\b'),                               # CA SIN 3-3-3
    re.compile(r'\b\d{4}(?:[\s\-]\d{3,6}){2,4}\b'),                           # cards 4-4-4-4 / 4-6-5 / 4-4-4-3
    re.compile(r'\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b'),  # IBAN (spaced or not)
    re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),                               # IPv4
    re.compile(r'(?<![\w:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![\w:])'),     # IPv6 full
    re.compile(r'(?<![\w:])(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}(?![\w:])'),  # IPv6 ::
    re.compile(r'\b[0-9a-fA-F]{2}(?:[:\-][0-9a-fA-F]{2}){5}\b'),              # MAC
    re.compile(r'\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b'),                              # UK number plate
    re.compile(r'\b[A-Z]{1,5}-?\d{5,}(?:-[A-Z0-9]{1,4})?\b'),                 # prefixed IDs: CID-410603, MED84969521
    re.compile(r'(?<![/\w.])\d{8,19}\b'),                                     # bare 8–19 digit numbers
    re.compile(r'(?<![\w\-])(?=(?:[A-Z]*\d){3})(?=(?:\d*[A-Z]){3})[A-HJ-NPR-Z0-9]{17}(?![\w\-])'),  # VIN
]

# Segmented alphanumeric IDs: 'ID-142-V', 'C371-9446-549', 'Of-54618', 'Yb37515-L', 'K02-7871-463-98'.
_SEG_ID_RE = re.compile(r'(?<![\w\-])[A-Z][A-Za-z0-9]{0,7}(?:-[A-Za-z0-9]{1,8}){1,4}(?![\w\-])')
_SEG_ID_SKIP_PREFIX = {   # standards and version codes — not identifiers of a person
    'ISO', 'IEC', 'RFC', 'IEEE', 'CVE', 'CWE', 'SOC', 'PCI', 'NIST', 'ANSI', 'DIN', 'FY',
    'COVID', 'SARS', 'UTF', 'GPT', 'HTTP', 'TLS', 'SSL', 'PEP', 'SHA'}


def _segmented_ids(text: str):
    for m in _SEG_ID_RE.finditer(text):
        v = m.group()
        digits = sum(ch.isdigit() for ch in v)
        if digits < 3 or not any(ch.isalpha() for ch in v):
            continue
        if v.split('-')[0].upper() in _SEG_ID_SKIP_PREFIX:
            continue
        if re.fullmatch(r'[A-Za-z]{1,3}\d?-(?:19|20)\d{2}', v):           # FY-2024, Q3-2024
            continue
        yield m


# All-caps segmented IDs with few digits: UID-BAYGBZ33, 0LP8-2EV4-JYMB.
_CAPS_SEG_ID_RE = re.compile(r'(?<![\w\-])[A-Z0-9]{2,12}(?:-[A-Z0-9]{2,12}){1,4}(?![\w\-])')
# Unsegmented all-caps codes: IDCHW1DHT0F8, X2635M974 (≥ 3 letter/digit switches).
_CAPS_CODE_RE = re.compile(r'(?<![\w\-])[A-Z0-9]{8,20}(?![\w\-])')


def _caps_ids(text: str):
    for m in _CAPS_SEG_ID_RE.finditer(text):
        v = m.group()
        core = v.replace('-', '')
        if len(core) < 8 or not re.search(r'\d', v) or sum(c.isalpha() for c in v) < 2:
            continue
        if re.match(r'\d{2,4}-\d{1,2}-\d{1,4}(?:T\d|$)', v):
            continue                                   # ISO dates / timestamps (1978-11-24T09)
        if v.split('-')[0] in _SEG_ID_SKIP_PREFIX or re.fullmatch(r'[\d\-]+', v):
            continue
        yield m
    for m in _CAPS_CODE_RE.finditer(text):
        v = m.group()
        if sum(c.isdigit() for c in v) < 2 or sum(c.isalpha() for c in v) < 2:
            continue
        switches = sum(a.isdigit() != b.isdigit() for a, b in zip(v, v[1:]))
        if switches < 3 or re.match(r'Q[1-4]FY', v) or re.match('|'.join(_SEG_ID_SKIP_PREFIX), v):
            continue
        yield m


def naive_uid_detect(text: str) -> list[dict]:
    """Structured identifiers + crypto wallets, with offsets."""
    out, seen = [], set()
    for pat in _UID_RES:
        for m in pat.finditer(text):
            if (m.start(), m.end()) in seen:
                continue
            if _is_currency(_prev_nonspace(text, m.start())):
                continue                                           # $43820950 is money
            _pref = re.match(r'[A-Za-z]+', m.group())
            if _pref and _pref.group().upper() in _SEG_ID_SKIP_PREFIX:
                continue                                           # ISO-27001, RFC-7231
            seen.add((m.start(), m.end()))
            out.append({'word': m.group(), 'entity_group': 'uid', 'start': m.start(), 'end': m.end()})
    for m in list(_segmented_ids(text)) + list(_caps_ids(text)):
        if (m.start(), m.end()) not in seen:
            seen.add((m.start(), m.end()))
            out.append({'word': m.group(), 'entity_group': 'uid', 'start': m.start(), 'end': m.end()})
    out += [{**c, 'entity_group': 'uid'} for c in naive_crypto_detect(text)
            if (c['start'], c['end']) not in seen]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §11c Username rules
# ═════════════════════════════════════════════════════════════════════════════
_USERNAME_TECH = {'sha256', 'sha512', 'sha1', 'md5', 'win32', 'win64', 'utf8', 'utf16', 'base64',
                  'ipv4', 'ipv6', 'mp3', 'mp4', 'h264', 'h265', 'x264', 'covid19', 'office365',
                  'windows10', 'windows11', 'python3', 'python2', 'es2015', 'es2020', 'gpt4',
                  'iphone15', 'iphone14', 'ps5', 'xbox360', 'fy2023', 'fy2024', 'q1', 'q2', 'q3', 'q4'}
_USERNAME_RES = [
    # keyword context: "username: jdoe92", "login=jsmith", "handle @jane"
    (re.compile(r"(?i:\b(?:user(?:[ _-]?name)?|login|handle|user[ _-]?id|account[ _-]?name|nick(?:name)?)\b)"
                r"[\"']?\s*[:=]\s*[\"']?@?([A-Za-z0-9][A-Za-z0-9._\-]{2,31})"), 1),
    (re.compile(r"(?<![\w@.])@([A-Za-z0-9_]{3,30})\b(?!\.\w)"), 1),                        # @handle
    (re.compile(r"(?<![\w@.])([A-Za-z][a-z]+[._][A-Za-z][a-z]+\d{0,4})\b(?![@.]\w)"), 1),   # jane.doe
    (re.compile(r"\b([A-Z][a-z]{1,15}\d{2,4})\b"), 1),                                     # Bernardo65
    (re.compile(r"\b([a-z]{3,15}[._]?\d{2,4})\b"), 1),                                     # aahana63
]
_TLDS = {'com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'uk', 'de', 'fr', 'info', 'biz', 'us',
         'eu', 'me', 'ai', 'app', 'dev', 'name', 'au', 'ca', 'in', 'nl', 'jp', 'cn', 'ru', 'br'}
_FILE_EXTS = {'csv', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'json', 'xml', 'png', 'jpg',
              'jpeg', 'gif', 'zip', 'log', 'md', 'py', 'js', 'html', 'htm', 'ppt', 'pptx', 'yaml',
              'yml', 'sql', 'exe', 'dll', 'tar', 'gz', 'mp3', 'mp4', 'env', 'cfg', 'ini'}
_KEY_AFTER_RE = re.compile(r'\s*=|["\']\s*:')      # field names: key=…, "key": — not chat "jsmith_92: hi"


def naive_username_detect(text: str) -> list[dict]:
    out, seen = [], set()
    for pat, g in _USERNAME_RES:
        for m in pat.finditer(text):
            v = m.group(g)
            if v.lower() in _USERNAME_TECH or v.lower() in {'user', 'login', 'name'}:
                continue
            if '.' in v and v.rsplit('.', 1)[1].lower() in _TLDS | _FILE_EXTS:
                continue                                       # gmail.com, example.org, notes.csv
            if g == 1 and pat is not _USERNAME_RES[0][0] and _KEY_AFTER_RE.match(text, m.end(g)):
                continue                                       # keys, not values: pw_attempt=…
            key = (m.start(g), m.end(g))
            if key not in seen:
                seen.add(key)
                out.append({'word': v, 'entity_group': 'username', 'start': key[0], 'end': key[1]})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §12 Phone numbers
# ═════════════════════════════════════════════════════════════════════════════
_PH_SP = r'[ \t\u00A0\u2009\u202F]'                      # space incl. NBSP / thin spaces
_PH_DASH = r'[.\-/\u2010-\u2015\u2212]'                  # . - / and unicode dashes / minus
_PH_SEP = rf'(?:{_PH_SP}?{_PH_DASH}{_PH_SP}?|{_PH_SP})'  # one separator, optional padding
_PH_UNIT = r'(?:\(\d{1,5}\)|\d)'                         # a digit or a "(area)" group

_PHONE_CANDIDATE_RE = re.compile(
    r'(?<![\w+\uFF0B])'                 # not glued to a word, or to a preceding '+'
    r'(?<!\d[.\-/])'                    # not continuing an earlier digit run (e.g. 12-345…)
    r'(?P<num>'
    r'(?:[+\uFF0B]' + _PH_SP + r'?)?'                  # optional international '+'
    + _PH_UNIT + r'(?:' + _PH_SEP + r'?' + _PH_UNIT + r'){5,24}'
    r')'
    r'(?P<ext>' + _PH_SP + r'*(?:ext\.?|extension|x|\#)' + _PH_SP + r'*\d{1,6})?'
    r'(?!\w)'                           # not glued to a following word/digit
    r'(?!' + _PH_DASH + r'\d)',         # nor continuing into another digit group
    re.IGNORECASE,
)
_PH_EXCLUDE = [
    re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$'),               # ISO date 2024-01-15
    re.compile(r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$'),             # 15/01/2024, 01.15.24
    re.compile(r'^(?:19|20)\d{2}\s?[-/\u2013]\s?(?:19|20)\d{2}$'),  # year range 2019-2024
    re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$'),                     # IPv4
    re.compile(r'^\d{3}-\d{2}-\d{4}$'),                           # US SSN
    re.compile(r'^\d{4}(?:[ -]?\d{4}){3}$'),                      # 16-digit card
    re.compile(r'^(?:(?:19|20)\d{2}[\s,/\-]+){1,}(?:19|20)\d{2}$'),  # list of years
]
_PH_CURRENCY = set('$€£¥₹₩₽¢')
PHONE_MIN_DIGITS = 7                 # shortest subscriber numbers (with or without +)
PHONE_MAX_DIGITS = 15                # E.164 maximum
PHONE_MIN_DIGITS_UNFORMATTED = 9     # bare digit runs need more evidence
PHONE_MAX_GLUED_DIGITS = 30          # two numbers separated only by a space match as ONE candidate


def naive_phone_detect(text: str) -> list[dict]:
    """International (+CC / 00CC), trunk-prefixed, parenthesised area codes, any
    space/dot/dash/slash grouping, unicode dashes/NBSPs, and extensions."""
    results, seen = [], set()
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        num = m.group('num')
        digits = sum(ch.isdigit() for ch in num)
        has_plus = num.lstrip()[:1] in '+\uFF0B'
        formatted = has_plus or '(' in num or bool(re.search(_PH_SEP, num))
        if digits < PHONE_MIN_DIGITS:
            continue
        if not formatted and not (PHONE_MIN_DIGITS_UNFORMATTED <= digits <= PHONE_MAX_DIGITS):
            continue          # bare digit runs: 9–15 digits; longer ones are UIDs/cards
        if formatted and digits > PHONE_MAX_GLUED_DIGITS:
            continue          # giant formatted runs are tables / IDs, not phones
        groups = re.findall(r'\d+', num)
        if len(groups) > 6 and digits / len(groups) < 2.5:
            continue          # "1 2 3 4 5 6 7 …" — a list of small numbers
        bare = num.strip()
        if any(p.match(bare) for p in _PH_EXCLUDE):
            continue
        j = m.start() - 1
        while j >= 0 and text[j] in ' \t\u00A0\u202F':
            j -= 1
        if j >= 0 and (text[j] in _PH_CURRENCY or unicodedata.category(text[j]) == 'Sc'):
            continue          # $1234567, ₪ 706511.38
        if re.fullmatch(r'\d{1,9}[.,]\d{2}', bare):
            continue          # plain decimal amount 706511.38
        if re.fullmatch(r'\d{1,3}\.\d{5,}', bare) or (m.start() > 0 and text[m.start() - 1] in ':-'):
            continue          # coordinates, fractional seconds
        key = (m.start(), m.end())
        if key not in seen:           # by position: a number that appears twice is flagged twice
            seen.add(key)
            results.append({'word': m.group().strip(), 'entity_group': 'phone',
                            'start': m.start(), 'end': m.end()})
    return results


# ═════════════════════════════════════════════════════════════════════════════
# §13 File paths
# ═════════════════════════════════════════════════════════════════════════════
_WIN_CH = r'[^\\/:*?"<>|\r\n\t ]'                    # legal Windows name char, no space
_WIN_DIR = rf'{_WIN_CH}+(?: {_WIN_CH}+){{0,2}}'      # directory name, ≤ 2 internal spaces
_WIN_TAIL = rf'(?:{_WIN_DIR}[\\/])*{_WIN_CH}*'       # dirs + optional final name
_NIX_CH = r'[\w.\-~@+%#=,]'                          # conservative POSIX segment chars
_NIX_DIR = rf'{_NIX_CH}+(?: {_NIX_CH}+){{0,2}}'      # "Old Files", "My Drive"
_NIX_TAIL = rf'(?:{_NIX_DIR}/)*{_NIX_CH}*'

_FILEPATH_RES = [
    re.compile(r'\bfile:///?[^\s"\'<>]+', re.I),                                  # file:// URI
    re.compile(rf'(?<![\w/\\])[A-Za-z]:[\\/]{_WIN_TAIL}'),                        # C:\…  C:/…
    re.compile(rf'(?<![\w\\])\\\\[\w.$\-]+\\{_WIN_TAIL}'),                        # \\server\share
    re.compile(rf'%[A-Za-z_][A-Za-z0-9_]*%[\\/]{_WIN_TAIL}'),                     # %APPDATA%\…
    re.compile(rf'(?<![\w.])\.{{1,2}}[\\/](?:{_WIN_TAIL})'),                      # ./x ../x .\x
    re.compile(rf'(?<![\w/])~[\w.\-]*/{_NIX_TAIL}'),                              # ~/x ~user/x
    re.compile(rf'(?<![\w$])\$\{{?[A-Za-z_][A-Za-z0-9_]*\}}?/{_NIX_TAIL}'),       # $HOME/x
    re.compile(rf'(?<![\w:/.\\~$%])/{_NIX_DIR}/{_NIX_TAIL}'),                     # /home/jsmith
    re.compile(r'(?<![\w:/.\\~$%@])(?:[\w\-]+[\\/])+[\w\-. ]*?[\w\-]\.[A-Za-z][A-Za-z0-9]{0,6}\b'),  # a/b/q3.xlsx
]
_FP_URL_RE = re.compile(r'[a-z][a-z0-9+.\-]*://', re.I)
_FP_DOMAIN_RE = re.compile(
    r'^(?:www\.)?[\w\-]+(?:\.[\w\-]+)*\.(?:com|org|net|edu|gov|io|co|uk|de|fr|au|ca|'
    r'info|biz|us|eu|me|ai|app|dev|ly|tv|nl|jp|cn|in|ru|br|es|it|ch|se|no|nz)(?:[/:]|$)', re.I)
_FP_DATE_RE = re.compile(r'^\d{1,4}/\d{1,2}/\d{1,4}$')
_FP_FRAC_RE = re.compile(r'^\d+/\d+$')
_FP_STRIP = '.,;:!?)]}\'"`>'


def naive_filepath_detect(text: str) -> list[dict]:
    """Return [{'word', 'entity_group': 'filepath', 'start', 'end'}] for path-like strings."""
    spans: list[tuple[int, int]] = []
    for pat in _FILEPATH_RES:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in _FP_STRIP:      # trailing sentence punctuation
                e -= 1
            val = text[s:e]
            if len(val) < 4 or not re.search(r'[\\/]', val):
                continue
            if not val.lower().startswith('file:'):
                if _FP_URL_RE.search(val) or _FP_URL_RE.search(text[max(0, s - 12):s + 3]):
                    continue
            if _FP_DOMAIN_RE.match(val) or _FP_DATE_RE.match(val) or _FP_FRAC_RE.match(val):
                continue
            spans.append((s, e))
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [{'word': text[s:e], 'entity_group': 'filepath', 'start': s, 'end': e}
            for s, e in merged]


# ── Registry: names used in the pipeline spec ('rule:<name>') ─────────────────
RULES = {
    'password': naive_password_detect,
    'street':   naive_street_detect,
    'email':    naive_email_detect,
    'name':     naive_name_detect,
    'org':      naive_org_detect,
    'uid':      naive_uid_detect,
    'username': naive_username_detect,
    'phone':    naive_phone_detect,
    'filepath': naive_filepath_detect,
}
