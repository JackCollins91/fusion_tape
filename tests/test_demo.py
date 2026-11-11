"""The benchmark scoring in demo.py: lining redacted text up with the original."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('demo', Path(__file__).resolve().parent.parent / 'demo.py')
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def shown(original, redacted):
    return ''.join(ch if v else '█' for ch, v in zip(original, demo.visible(original, redacted)))


def test_mask_marks_what_each_marker_replaced():
    assert shown('Call Jane Doe on 555 0100.', 'Call [[NAME]] Doe on [[TELEPHONE]].') == \
        'Call ████ Doe on ████████.'


def test_mask_with_markers_at_both_ends_and_combined_tags():
    assert shown('john_smith logged in as Jane', '[[NAME/USERNAME]] logged in as [[NAME]]') == \
        '██████████ logged in as ████'


def test_ambiguous_piece_is_not_counted_as_visible():
    text = 'between Will and Sons and Ziemann, Legros and Bogan on Monday'
    assert shown(text, 'between [[ORGANISATION]] and [[ORGANISATION]] on Monday') == \
        'between ' + '█' * 43 + ' on Monday'


def test_no_markers_means_everything_visible():
    assert shown('nothing here', 'nothing here') == 'nothing here'


def test_changed_text_outside_markers_is_an_error():
    with pytest.raises(ValueError):
        demo.visible('Call Jane now', 'Ring [[NAME]] now')


def test_leak_needs_three_visible_letters_or_digits_and_ignores_marker_type():
    text = 'Ms Jane Doe, user jdoe'
    pii = [(0, 2, 'name'), (3, 7, 'name'), (8, 11, 'name'), (18, 22, 'username')]
    assert demo.score(text, pii, 'Ms [[USERNAME]] Doe, user [[NAME]]') == [
        ('name', 'Jane', False),        # covered, by a marker of another domain: not a leak
        ('name', 'Doe', True),
        ('username', 'jdoe', False),
    ]                                   # 'Ms' is too short to score
