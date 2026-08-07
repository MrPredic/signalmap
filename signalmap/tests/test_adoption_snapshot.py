"""Adoption snapshot — the merge must never lose or double-count a day.

GitHub's traffic API keeps 14 days; the store is the only durable record, so
the parsing and merging logic is tested offline against fixture payloads. No
network, no ``gh``: the fetch layer is deliberately not exercised here.
"""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "adoption_snapshot.py"


def _load():
    spec = importlib.util.spec_from_file_location("adoption_snapshot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ads = _load()


def _snap(pypi=None, clones=None, views=None, ci=None, repo=None):
    return {
        "pypi_downloads": pypi or {},
        "clones": clones or {},
        "views": views or {},
        "ci_runs": ci or {},
        "referrers": [],
        "paths": [],
        "repo": repo or {},
    }


def test_pypi_days_drops_mirror_rows():
    payload = {"data": [
        {"category": "with_mirrors", "date": "2026-08-05", "downloads": 999},
        {"category": "without_mirrors", "date": "2026-08-05", "downloads": 234},
    ]}
    assert ads.pypi_days(payload) == {"2026-08-05": 234}


def test_traffic_days_keys_by_date():
    payload = {"views": [
        {"timestamp": "2026-08-05T00:00:00Z", "count": 10, "uniques": 9},
    ]}
    assert ads.traffic_days(payload, "views") == {
        "2026-08-05": {"count": 10, "uniques": 9}
    }


def test_runs_days_counts_runs_per_day():
    payload = {"workflow_runs": [
        {"created_at": "2026-08-05T16:00:00Z"},
        {"created_at": "2026-08-05T17:00:00Z"},
        {"created_at": "2026-08-04T09:00:00Z"},
    ]}
    assert ads.runs_days(payload) == {"2026-08-05": 2, "2026-08-04": 1}


def test_missing_sources_parse_to_empty():
    for parse in (ads.pypi_days, ads.runs_days):
        assert parse(None) == {}
    assert ads.traffic_days(None, "clones") == {}


def test_merge_keeps_days_that_fell_out_of_the_window():
    old = {"2026-07-20": 5, "2026-08-01": 1}
    new = {"2026-08-01": 1, "2026-08-05": 234}
    assert ads.merge_series(old, new) == {
        "2026-07-20": 5, "2026-08-01": 1, "2026-08-05": 234
    }


def test_rerun_same_day_overwrites_instead_of_adding():
    store = ads.update({}, _snap(pypi={"2026-08-05": 100}), "2026-08-05T07:00:00+00:00")
    store = ads.update(store, _snap(pypi={"2026-08-05": 234}), "2026-08-05T19:00:00+00:00")
    assert store["pypi_downloads"] == {"2026-08-05": 234}


def test_update_records_deliberate_acts_and_timestamp():
    store = ads.update(
        {},
        _snap(repo={"stargazers_count": 3, "forks_count": 1,
                    "subscribers_count": 2, "open_issues_count": 0}),
        "2026-08-06T05:21:00+00:00",
    )
    assert store["marks"]["2026-08-06"]["stars"] == 3
    assert store["last_snapshot"] == "2026-08-06T05:21:00+00:00"


def test_failed_repo_fetch_writes_no_mark():
    store = ads.update({}, _snap(), "2026-08-06T05:21:00+00:00")
    assert store["marks"] == {}


def test_report_shows_ci_runs_next_to_clones():
    store = ads.update(
        {},
        _snap(pypi={"2026-07-30": 118},
              clones={"2026-07-30": {"count": 69, "uniques": 26}},
              views={"2026-07-30": {"count": 10, "uniques": 9}},
              ci={"2026-07-30": 8}),
        "2026-07-30T20:00:00+00:00",
    )
    line = [ln for ln in ads.report(store).splitlines() if ln.startswith("2026-07-30")][0]
    assert "118" in line and "69/26" in line and "10/9" in line and line.rstrip().endswith("8")


def test_store_roundtrips_through_json(tmp_path):
    store = ads.update({}, _snap(pypi={"2026-08-05": 234}), "2026-08-05T19:00:00+00:00")
    path = tmp_path / "adoption.json"
    path.write_text(json.dumps(store))
    assert ads.main(["--report", "--store", str(path)]) == 0
