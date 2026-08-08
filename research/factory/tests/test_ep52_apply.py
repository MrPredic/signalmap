from datetime import datetime

from ep52_apply import planned_times


def test_ep52_times_are_fixed_from_documented_onset():
    onset = datetime.fromisoformat("2026-07-15T18:30")
    overflow = datetime.fromisoformat("2026-07-15T00:51")
    out = planned_times(onset, overflow)
    assert [x["offset_h"] for x in out] == [1, 6, 12]
    assert out[0]["pre"] == "2026-07-15T17:30:00"
    assert out[-1]["pre"] == "2026-07-15T06:30:00"


def test_ep52_rejects_overflow_after_onset():
    onset = datetime.fromisoformat("2026-07-15T18:30")
    overflow = datetime.fromisoformat("2026-07-15T19:00")
    try:
        planned_times(onset, overflow)
    except ValueError as exc:
        assert "overflow" in str(exc).lower()
    else:
        raise AssertionError("invalid HVO timing was accepted")
