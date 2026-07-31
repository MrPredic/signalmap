"""`signalmap discover` — the confound-ablation claim, checked against its own
ground truth. If the naive mode stops being wrong, the demo stops proving that
confound removal is necessary at all.
"""
from signalmap import discover


def test_evaluate_keeps_the_true_coupling_and_rejects_the_confounded_pair():
    v = discover.evaluate(seed=0)
    assert v["true_coupling_found"]
    assert v["confound_rejected"]


def test_the_confounded_pair_looks_strong_raw_and_collapses_once_adjusted():
    v = discover.evaluate(seed=0)
    assert v["confound_raw_corr"] > 0.3          # the trap a naive scan falls into
    assert v["confound_adj_corr"] < v["confound_raw_corr"] / 2


def test_evaluate_is_stable_across_seeds():
    for seed in (1, 2, 3):
        v = discover.evaluate(seed=seed)
        assert v["true_coupling_found"] and v["confound_rejected"], f"seed {seed}"


def test_naive_mode_reports_more_survivors_than_the_adjusted_run(capsys):
    naive = discover.run(naive=True, n=800, seed=0)
    adjusted = discover.run(n=800, seed=0)
    n_naive = sum(r["survives"] for r in naive)
    n_adj = sum(r["survives"] for r in adjusted)
    assert n_naive > n_adj, "naive mode no longer demonstrates the confound trap"


def test_run_prints_a_verdict_line_per_pair(capsys):
    results = discover.run(n=800, seed=0)
    out = capsys.readouterr().out
    assert "confound-adjusted" in out
    assert "survivors" in out
    for r in results:
        assert f"{r['a']}–{r['b']}" in out


def test_cli_defaults_to_temp_as_the_confound(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["signalmap-discover", "--n", "800"])
    discover.main()
    assert "given: ['temp']" in capsys.readouterr().out
