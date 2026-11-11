"""
Sanity checks for every rule in fusion_tape.rules.
If a rule is edited, these show whether it still behaves as evaluated.
"""
import pytest

from fusion_tape import rules
from fusion_tape.normalise import drop_first_letter, map_span_back, normalise_for_names


# ── §5 Password ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize('token, expect', [
    ('Xk9#mLpQ2!', True), ('zP$4wR!8vT', True), ('abc12345', False), ('2024-01-15', False),
    ('https://example.com/path?v=123', False), ('user123@domain.com', False), ('$1,234.56', False),
    ('correct horse battery staple.', False), ('bZYeFcwGKIb2', True), ('JHhsDZTkkGpr', True),
    ('₪ 706511.38', False), ('MacBookPro', False), ('iPhone15', False), ('mRNA-1273', False),
    ('session=d6895e3a-747c-4db7-8b6d-5877f4c16136', False), ('ua=Mozilla/5.0', False),
    ('**$82.50**', False), ('pw_attempt=+8k33H1jMb18Zqd_6(', True), ('pw_attempt=rtptEsaxl5', True),
])
def test_password(token, expect):
    assert bool(rules.naive_password_detect(token)) == expect


# ── §8b Normaliser + drop-first view ──────────────────────────────────────────
@pytest.mark.parametrize('src, expect', [
    ('john_smith', 'john smith'), ('jane.doe92', 'jane doe 92'), ('JohnSmith', 'John Smith'),
    ('JSmith', 'J Smith'), ('MaryJane_O.Connor', 'Mary Jane O Connor'), ('Bernardo65', 'Bernardo 65'),
    ('Ludwig_Halvorson53', 'Ludwig Halvorson 53'), ('JoséÁlvarez', 'José Álvarez'),
    ('Pi is 3.14. Mr. Smith left.', 'Pi is 3.14. Mr. Smith left.'), ('USA', 'USA'),
    ('NASAGov', 'NASA Gov'), ('Mary-Jane', 'Mary Jane'), ('jean-luc.picard', 'jean luc picard'),
    ('Hernandez-Padilla', 'Hernandez Padilla'),
    ('1984-07-16T07:20:36.5', '1984-07-16T07:20:36.5'),
    ('session=d6895e3a-747c-4db7', 'session=d6895e3a-747c-4db7'),
    ('DEBUG v6 peer', 'DEBUG v6 peer'), ('1LEB5XXi62uPk5XtXa', '1LEB5XXi62uPk5XtXa'),
])
def test_normalise_for_names(src, expect):
    assert normalise_for_names(src)[0] == expect


def test_normalised_span_maps_back():
    t = 'Contact jane.doe92 or JohnSmith today'
    n, m = normalise_for_names(t)
    s = n.index('John Smith')
    a, b = map_span_back(m, s, s + len('John Smith'))
    assert t[a:b] == 'JohnSmith'


def test_drop_first_letter():
    vocab = {'reset', 'the', 'vpn', 'for', 'temp', 'pass', 'and', 'logged', 'in', 'nothing',
             'else', 'was', 'reported'}
    is_vocab = lambda w: w.lower() in vocab
    assert drop_first_letter('reset the vpn for xnunez? temp pass', is_vocab)[0] == \
        'reset the vpn for Nunez? temp pass'
    assert drop_first_letter('jsmith92 and lelliott logged in', is_vocab)[0] == 'Smith and Elliott logged in'
    assert drop_first_letter('nothing else was reported', is_vocab)[0] == 'nothing else was reported'


# ── §8c Names ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('text, expect', [
    ('Sincerely, Ms.', 'Ms.'), ('evaluating Mrs.Auer. We', 'Mrs.Auer'),
    ('Dr. Ada Sterling (SSN', 'Dr. Ada Sterling'), ('Hello Skyler, we', 'Skyler'), ('Dear Team, we', None),
])
def test_name_rules(text, expect):
    got = [h['word'] for h in rules.naive_name_detect(text)]
    assert (expect in got) if expect else not got


# ── §9b Organisations ─────────────────────────────────────────────────────────
@pytest.mark.parametrize('text, expect', [
    ('Franey - Boyle offers', 'Franey - Boyle'), ('details for Mcbride-Jones: SSN', 'Mcbride-Jones'),
    ('Consignee: Johnson Inc Description', 'Johnson Inc'),
    ('at Stracke, Rohan and Steuber, I', 'Stracke, Rohan and Steuber'), ('the company was small', None),
])
def test_org_rules(text, expect):
    got = [h['word'] for h in rules.naive_org_detect(text)]
    assert (expect in got) if expect else not got


# ── §10 IDs and crypto ────────────────────────────────────────────────────────
@pytest.mark.parametrize('value, expect', [
    ('ID-142-V', True), ('C371-9446-549', True), ('O6613-Ad', True), ('Of-54618', True),
    ('Yb37515-L', True), ('K02-7871-463-98', True), ('WK39-2027', True), ('LIC-M869061', True),
    ('UENJDJUTX2BP01Y1N', True), ('1HGCM82633A004352', True), ('UID-BAYGBZ33', True),
    ('UID-BAIEHLK6', True), ('0LP8-2EV4-JYMB', True), ('IDCHW1DHT0F8', True), ('X2635M974', True),
    ('WINDOWS10', False), ('MP3PLAYER', False), ('1978-11-24T09', False), ('COVID-19', False),
    ('SARS-CoV-2', False), ('mRNA-1273', False), ('ISO-27001', False), ('CVE-2024-1234', False),
    ('FY-2024', False), ('Mcbride-Jones', False), ('2024-01-15', False),
])
def test_segmented_ids(value, expect):
    got = any(h['word'] == value for h in rules.naive_uid_detect(f'Ref: {value}, please.'))
    assert got == expect


@pytest.mark.parametrize('value, expect', [
    ('3SPRsod2g8wBe7jfCnb74g1BwSBwatvWEAv', True), ('MAZnandVcoJZ71AeyrBrjK9LHW', True),
    ('bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq', True),
    ('0x52908400098527886E0F7030069857D2E4169EE7', True),
    ('ltc1qg82tpnmjw0wu9lpvf0pz2zm3k8vjzdlmx6k0p6', True),
    ('Antidisestablishmentarianism', False), ('9e2beea1aaff5b078f23bcc7cf52159614cd122a', False),
])
def test_crypto(value, expect):
    assert bool(rules.naive_crypto_detect(f'send to {value} now')) == expect


# ── §11c Usernames ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('text, expect', [
    ('Use Bernardo65 as the reference', 'Bernardo65'), ('maintain Treva.Koss confidentiality', 'Treva.Koss'),
    ('"nonce": "parth_2708"', 'parth_2708'), ('username: kevinmyers', 'kevinmyers'),
    ('mail Kirk.Abshire@gmail.com now', None), ('hash is sha256 here', None),
    ('pw_attempt=1E4FKCnp4d', None), ('{"api_hash": "x"}', None), ('see notes.csv', None),
    ('user_name: jdoe_92', 'jdoe_92'), ('[09:14] jsmith_92: hey', 'jsmith_92'),
])
def test_username_rules(text, expect):
    got = [h['word'] for h in rules.naive_username_detect(text)]
    assert (expect in got) if expect else not got


# ── §12 Phones ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('value, expect', [
    ('+44 20 7946 0958', True), ('+44 (0)20 7946 0958', True), ('(555) 123-4567', True),
    ('+1-555-123-4567 ext. 89', True), ('0412 345 678', True), ('+61 2 9876 5432', True),
    ('+33 1 23 45 67 89', True), ('+49 (0)30 1234-5678', True), ('+91-98765-43210', True),
    ('+86 138 0013 8000', True), ('03-1234-5678', True), ('+7 (495) 123-45-67', True),
    ('+55 11 91234-5678', True), ('00 44 20 7946 0958', True), ('(041) 7342-6271', True),
    ('+45 29.718-5733', True), ('5551234567', True), ('555\u2013123\u20134567', True),
    ('2024-01-15', False), ('15/01/2024', False), ('192.168.1.100', False), ('123-45-6789', False),
    ('4111 1111 1111 1111', False), ('$1234567', False), ('2019-2024', False), ('Order #12345', False),
    ('1 2 3 4 5 6 7 8 9 10 11 12', False), ('v10.0.19045', False), ('₪ 706511.38', False),
    ('706511.38', False), ('at 07:20:18.886663', False), ('36.534664', False),
])
def test_phone(value, expect):
    assert bool(rules.naive_phone_detect(f'Call {value} today.')) == expect


# ── §13 File paths ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('value, expect', [
    (r'C:\Users\jsmith\Documents\report.docx', True), (r'C:\Users\Jane Doe\Desktop\cv final.pdf', True),
    (r'\\fileserver\hr$\payroll\2024.xlsx', True), (r'%USERPROFILE%\Desktop\notes.txt', True),
    ('/home/jsmith/.ssh/id_rsa', True), ('~/Documents/tax-2023.pdf', True),
    ('$HOME/.aws/credentials', True), (r'..\..\data\users.csv', True),
    ('Documents/Reports/q3_summary.xlsx', True), ('file:///C:/Users/jane/report.docx', True),
    ('https://example.com/path/to/page.html', False), ('example.com/index.html', False),
    ('2024/01/15', False), ('and/or', False), ('30 km/h', False), ('24/7', False),
    ('s3://bucket/key.json', False),
])
def test_filepath(value, expect):
    hits = rules.naive_filepath_detect(f'See {value} for details.')
    got = any(h['word'] == value for h in hits) if expect else bool(hits)
    assert got == expect


def test_email():
    assert [h['word'] for h in rules.naive_email_detect('mail jane.doe@example.co.uk now')] == \
        ['jane.doe@example.co.uk']


def test_street():
    words = [h['word'] for h in rules.naive_street_detect('Ship to 42 Oak Street, Apt. 217 today')]
    assert '42 Oak Street' in words and 'Apt. 217' in words
