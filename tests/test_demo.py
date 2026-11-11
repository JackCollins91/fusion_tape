"""The benchmark scoring in demo.py: lining redacted text up with the original."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('demo', Path(__file__).resolve().parent.parent / 'demo.py')
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def shown(original, redacted):
    return ''.join(ch if v else '█' for ch, v in zip(original, demo.visible_mask(original, redacted)))


def test_mask_marks_what_each_marker_replaced():
    assert shown('Call Jane Doe on 555 0100.', 'Call [[NAME]] Doe on [[TELEPHONE]].') == \
        'Call ████ Doe on ████████.'


def test_mask_with_markers_at_both_ends_and_combined_tags():
    assert shown('john_smith logged in as Jane', '[[NAME/USERNAME]] logged in as [[NAME]]') == \
        '██████████ logged in as ████'


def test_no_markers_means_everything_visible():
    assert shown('nothing here', 'nothing here') == 'nothing here'


def test_changed_text_outside_markers_is_an_error():
    with pytest.raises(ValueError):
        demo.visible_mask('Call Jane now', 'Ring [[NAME]] now')


def test_leak_needs_three_visible_letters_or_digits_and_ignores_marker_type():
    text = 'Ms Jane Doe, user jdoe'
    gt = [{'label': 'PREFIX', 'start': 0, 'end': 2}, {'label': 'FIRSTNAME', 'start': 3, 'end': 7},
          {'label': 'LASTNAME', 'start': 8, 'end': 11}, {'label': 'USERNAME', 'start': 18, 'end': 22},
          {'label': 'JOBTITLE', 'start': 13, 'end': 17}]
    scored = demo.score_text(text, gt, 'Ms [[USERNAME]] Doe, user [[NAME]]')
    assert [(v['value'], v['domain'], v['leaked']) for v in scored] == [
        ('Jane', 'name', False),        # covered, by a marker of another domain: not a leak
        ('Doe', 'name', True),
        ('jdoe', 'username', False),
    ]                                   # 'Ms' is too short to score; JOBTITLE is not one of the nine domains
