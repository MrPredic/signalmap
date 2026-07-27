"""Ingest hardening for text recordings (product finding, Jul 13): real ECN
.txt files carry a UTF-8 BOM, a tab-separated header line, CRLF endings, and
RAGGED rows (most rows miss the last column, leaving a trailing tab) — plain
np.genfromtxt refuses them. The loader must read what is readable, count what
it skips, and fail closed when the requested column is mostly absent."""
import numpy as np
import pytest

from signalmap.distill import _read_recording, load_bank


def _ecn_like(path, n_rows=64, bom=True, header=True, ragged=True):
    """Reproduce the observed ECN file shape byte-for-byte in miniature."""
    lines = []
    if header:
        lines.append("Time (s)\tWE(1).Potential (V)\tOCP value (V)")
    rng = np.random.default_rng(0)
    third = lambda i: f"\t{rng.standard_normal():.6f}" if (not ragged or i == 0) else "\t"
    for i in range(n_rows):
        lines.append(f"{0.27 * i:.6f}\t{np.sin(0.1 * i) + 0.01 * i:.6f}{third(i)}")
    data = "\r\n".join(lines) + "\r\n"
    raw = data.encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def test_ecn_shaped_txt_column1(tmp_path):
    p = tmp_path / "FeCl3_0_1M.txt"
    _ecn_like(p)
    a = _read_recording(str(p), column=1)
    assert a.shape == (64,), "all data rows must survive; header must not"
    assert np.isfinite(a).all()
    assert a[0] == pytest.approx(np.sin(0.0))


def test_ecn_shaped_txt_column0_time(tmp_path):
    p = tmp_path / "FeCl3_0_1M.txt"
    _ecn_like(p)
    a = _read_recording(str(p), column=0)
    assert a.shape == (64,)
    assert a[1] == pytest.approx(0.27)


def test_mostly_missing_column_fails_closed(tmp_path):
    """Column 2 exists on only the first data row -> refusing beats returning
    a signal that silently dropped 98% of its samples."""
    p = tmp_path / "FeCl3_0_1M.txt"
    _ecn_like(p)
    with pytest.raises(SystemExit, match="column"):
        _read_recording(str(p), column=2)


def test_leading_missing_rows_still_fail_closed(tmp_path):
    """Fail-closed must depend on the bad FRACTION, not row order: a column
    that is empty for a leading block of data rows (sensor warm-up / wrong
    --column / truncated file) must trip the guard just like trailing gaps.
    A leading row that has a number in ANOTHER column is data, not a header."""
    p = tmp_path / "warmup.csv"
    lines = ["t,v"] + [f"{i}," for i in range(60)] + [f"{i},{i * 0.5}" for i in range(40)]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="60%|column"):
        _read_recording(str(p), column=1)


def test_all_junk_file_fails_closed(tmp_path):
    p = tmp_path / "junk.txt"
    p.write_text("no numbers here\nstill none\n")
    with pytest.raises(SystemExit):
        _read_recording(str(p), column=0)


def test_nonfinite_tokens_are_skipped_within_tolerance(tmp_path):
    p = tmp_path / "a.txt"
    rows = [f"{v:.3f}" for v in np.linspace(0, 1, 100)]
    rows[50] = "nan"
    p.write_text("\n".join(rows) + "\n")
    a = _read_recording(str(p), column=0)
    assert a.shape == (99,)
    assert np.isfinite(a).all()


def test_csv_with_bom_and_header(tmp_path):
    p = tmp_path / "b.csv"
    body = "t,v\n" + "\n".join(f"{i},{i * 0.5}" for i in range(32)) + "\n"
    p.write_bytes(b"\xef\xbb\xbf" + body.encode())
    a = _read_recording(str(p), column=1)
    assert a.shape == (32,)
    assert a[2] == pytest.approx(1.0)


def test_plain_single_column_txt_unchanged(tmp_path):
    """Backward compat: the clean files every existing bank uses."""
    p = tmp_path / "c.txt"
    x = np.sin(np.linspace(0, 20, 300))
    p.write_text("\n".join(f"{v:.9f}" for v in x) + "\n")
    a = _read_recording(str(p), column=0)
    assert np.allclose(a, x, atol=1e-8)


def test_load_bank_on_ecn_shaped_files(tmp_path):
    """Through the bank loader: enough rows for one 1024-window per file."""
    for cls in ("A", "B"):
        for r in range(2):
            _ecn_like(tmp_path / f"{cls}_{r}.txt", n_rows=1100)
    bank = load_bank(str(tmp_path), label_by="prefix", column=1)
    assert bank.n_recordings == 4
    assert bank.classes == ["A", "B"]
    assert all(len(w) == 1024 for w in bank.windows)
