"""Tests for the artifact upload progress callback.

The SDK hands an upload ``progress_cb`` a byte increment since 6.74.16 and passed the
multipart encoder monitor before that, so the callback has to accept both. Getting this
wrong killed a finished training right at the upload:

    AttributeError: 'int' object has no attribute 'bytes_read'

Only needs the `supervisely` package. The helper is loaded from its file rather than
imported as `src.upload_progress`, because `tests/conftest.py` puts `serve/` on sys.path
and both subapps ship a package called `src`.

Run with:  python -m pytest tests/test_upload_progress.py -v
"""

import importlib.util
from pathlib import Path

import pytest
import supervisely as sly

HELPER = Path(__file__).parents[1] / "train" / "src" / "upload_progress.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("upload_progress", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_upload_monitor = load_helper().make_upload_monitor

TOTAL = 30000
CHUNKS = [8192, 8192, 8192, 5424]


class _Pbar:
    """Stands in for the widget progress bar, which needs a running app."""

    def __init__(self):
        self.n = 0

    def update(self, count):
        self.n += count


class _Monitor:
    """The multipart encoder monitor the SDK used to pass."""

    def __init__(self, total):
        self.bytes_read = 0
        self.len = total

    def advance(self, count):
        self.bytes_read += count
        return self


@pytest.fixture
def progress():
    return sly.Progress("Uploading train artifacts", total_cnt=TOTAL, is_size=True)


def test_delta_increments_reach_the_total(progress):
    """The current contract: the callback is fed a byte increment."""
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)

    for chunk in CHUNKS:
        report(chunk)

    assert progress.current == TOTAL
    assert pbar.n == TOTAL


def test_encoder_monitor_reaches_the_total(progress):
    """The contract of SDK versions before 6.74.16, still worth accepting."""
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)
    monitor = _Monitor(TOTAL)

    for chunk in CHUNKS:
        report(monitor.advance(chunk))

    assert progress.current == TOTAL
    assert pbar.n == TOTAL


def test_monitor_reports_cumulative_and_delta_reports_increments(progress):
    """The two conventions differ in meaning, so mixing them up doubles the count."""
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)
    monitor = _Monitor(TOTAL)

    report(monitor.advance(8192))
    assert progress.current == 8192

    report(8192)  # an increment on top of what the monitor already reported
    assert progress.current == 16384


def test_progress_bar_never_goes_backwards(progress):
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)
    seen = []

    for chunk in CHUNKS:
        report(chunk)
        seen.append(pbar.n)

    assert seen == sorted(seen)
    assert seen[-1] == TOTAL


def test_unknown_total_is_taken_from_the_monitor():
    """With total_cnt=0 the size is unknown, and the monitor is the only source of it."""
    progress = sly.Progress("Uploading train artifacts", total_cnt=0, is_size=True)
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)

    report(_Monitor(TOTAL).advance(8192))

    assert progress.total == TOTAL
    assert progress.current == 8192


def test_a_finished_upload_is_reported_exactly_once(progress):
    """A zero sized increment can arrive at the end of the body."""
    pbar = _Pbar()
    report = make_upload_monitor(progress, pbar)

    for chunk in CHUNKS:
        report(chunk)
    report(0)

    assert progress.current == TOTAL
    assert pbar.n == TOTAL
