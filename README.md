<h1 align="center">SignalMap</h1>
<p align="center"><b>Unsupervised condition monitoring for any recorded signal — with verdicts you can check without trusting us.</b></p>
<p align="center">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
<img alt="Status" src="https://img.shields.io/badge/status-0.5.3-orange">
</p>

You give it healthy data. It learns what healthy looks like and flags deviations.
No fault labels, no per-sensor tuning. Vibration, acoustics, current — anything
you can record.

The part we care most about: every verdict it produces is a **signed receipt**
that a stranger can verify offline with a script that imports nothing from this
project. And where the method cannot honestly decide, it says **REFUSED** rather
than guessing.

## Install

```bash
python3 -m pip install 'signalmap[all]'   # quote the brackets so zsh does not glob them
signalmap plugins                         # confirm the install
```

`[distill]` alone is enough for `distill` / `fit` / `monitor` on `.npy` and
`.csv` banks. Parquet needs `pyarrow`, which `[all]` includes.

## Start here: does it work on *your* recordings?

Before anything else, ask the tool. Put your recordings in one directory named
`normal_*.wav` and `anomaly_*.wav` — a handful of known-bad ones is enough —
and run:

```bash
signalmap checkup --bank recordings/
```

```text
SEPARATES — AUC 0.335, 95% CI [0.240, 0.428] over 169 held-out recordings

  Direction: INVERTED — your faults score LOWER than healthy.
  That is a real pattern, not a bug: faults here are more stereotyped than
  normal operation. A detector that assumed 'far from healthy means faulty'
  would be worse than useless on this data.

  ALARM READY — cut 2.287 catches 53% of faulty recordings at 18% false alarms.
```

It fits on half your healthy recordings, scores the held-out half plus your
faulty ones, and answers `SEPARATES` or `REFUSED` with a confidence interval.
It will tell you it does not work on your data if it does not — that is the
whole point, and it takes a minute instead of a week.

## The monitoring flow

```bash
# 1. recordings -> frames. WAV, CSV and NPY all work.
signalmap ingest-file healthy.wav --label normal        --out data.parquet
signalmap ingest-file faulty.wav  --label ANOMALY_pump  --out data.parquet

# 2. fit on the healthy rows only, then watch
signalmap fit     --dataset data.parquet --healthy-label normal --out detector.pt
signalmap monitor --source replay --dataset data.parquet --detector detector.pt
```

`fit` never sees a fault label. `examples/` reproduces this end to end on
public bearing data.

## What it actually does, and where it stops

On CWRU bearing data, fitting on 945 healthy frames and monitoring 1183 frames
catches **238/238 faults with 2/945 false alarms**, fully unsupervised.

That number does not transfer, and we measured it rather than assuming it. On
nine domains under one frozen recipe — MIMII, Paderborn, MAFAULDA and two
constructed controls — the shipped alarm's gap between faulty and healthy is
**not above zero in eight of them**, and in six it **never fires at all**.

Worse, the direction is not even stable. On MIMII valve and Paderborn phase
current the detector ranks anomalies as *more normal than normal* (AUC 0.2786
and 0.3172, bootstrap CI clear of 0.5), while on MIMII slider it ranks them the
expected way (0.8254). There is a theorem behind that, and it applies to any
detector that sees only healthy data:

> The score is a function of the healthy distribution alone. Hold the fitted
> detector fixed, vary only the never-observed anomaly distribution, and the
> AUC attains every value in [0, 1].

So "far from healthy means faulty" is an assumption, not a result.
**[Read the study →](study/)** — preregistered before any data was touched,
nine domains, signed receipts, ten amendments including our own errors.

## What changed because of it

The detector no longer assumes the direction. It starts out refusing:

```python
det = DistilledDetector.fit(spec, healthy_windows)
det.decide(w).verdict            # 'REFUSED' — direction unknown

v = det.calibrate_direction(anchor_windows, labels, groups=recording_ids)
v.sign, v.ci_lo, v.ci_hi         # +1 or -1, only when the CI clears 0.5
det.decide(w).verdict            # now 'ALARM' or 'QUIET'
```

A handful of labelled anchors settles the direction. Without them, REFUSED is
the correct answer, and `groups` makes the bootstrap resample recordings rather
than windows — twenty windows cut from one signal are one observation.

## Verdicts you can check without us

Every verdict-producing command writes a signed JSON receipt: the claim, the
verdict (`INCLUDED` / `EXCLUDED` / `PASS` / `REFUSED`), the evidence, the sha256
of every input, and an Ed25519 signature. The signing key stays on your machine;
only the public key travels.

The verifier is one file that imports **nothing** from signalmap — stdlib plus
`cryptography`:

```bash
signalmap corpus --out receipts/     # the shipped verdicts

curl -sO https://raw.githubusercontent.com/MrPredic/signalmap/main/tools/verify_receipt.py
pip install cryptography
python3 verify_receipt.py receipts/cwru_rqa.receipt.json
# PASS — verdict INCLUDED, integrity only (NOT authenticity)

python3 verify_receipt.py receipts/cwru_rqa.receipt.json --pubkey <hex>
# PASS — verdict INCLUDED, authentic (pinned key)
```

Flip one byte in a receipt and verification fails. That does not make a verdict
*true* — it makes it attributable and tamper-evident. Without `--pubkey` you get
integrity only, and the tool says so rather than letting you misread it.

The verifier is hardened against 23 concrete attacks on the gap between the
bytes that get signed and the fields a human reads: duplicate JSON keys,
`NaN`/`Infinity` literals, integers past 2⁵³, unpaired surrogates, non-UTF-8
input, and a pinned-key check that used to compare transcriptions rather than
key bytes.

## distill: features chosen with a receipt

```bash
signalmap distill --bank recordings/ --label-by prefix --out artifacts/spec.json
```

Searches a compositional feature grammar for the few programs that separate
*your* recordings, under a capacity gate (`budget = C × n_recordings`). Every
run emits a gauntlet receipt: leakage-free nested LOGO accuracy, a
group-permutation p-value, a label-shuffle null, and cost per window.

Premium families (`--premium rqa,coherence`) run as opt-in challengers against
the distilled base and enter `spec.json` only on a CI-solid win. Of the eight
preregistered verdicts shipped in this repo, **six are exclusions** — the
mechanism refuses far more often than it admits.

## Live sources and where the output goes

The batch flow above is not the only one — `signalmap plugins` lists what is
wired in:

```text
source   : mic, mqtt, replay, sim
sink     : parquet, qdrant, questdb, stdout
```

So a microphone or an MQTT topic can feed `monitor` directly instead of a
parquet file, and scores can land in QuestDB for trend analysis rather than
scrolling past in a terminal:

```bash
signalmap monitor --source mic  --detector detector.pt          # live audio
signalmap monitor --source mqtt --broker tcp://plant:1883 --detector detector.pt
```

There is no SCADA or e-mail alerting adapter yet. Scores go to a sink; wiring
a sink to your alerting is currently your job, and a contributed sink is the
most useful thing anyone could add.

## More

- [`study/`](study/) — the sign-identifiability study: preregistration, theory,
  receipts, and the bank manifests with the sha256 of all 7623 recordings.
- [`examples/`](examples/) — reproduce the CWRU result and the acoustic and
  electrical recipes.
- `signalmap plugins` — sources, sinks, featurizers and models are all pluggable.
- Model artifacts (`.pt`) are a trust boundary: loaded with
  `weights_only=True`, arbitrary pickle objects rejected.

Experimental work — cross-modal discovery, compositional search, capture
adapters for salvaged hardware — lives behind its own flags and is labelled as
experimental where it appears. Nothing on this page depends on it.

## Contributing

Bug reports and reproductions are the most useful contributions: if a verdict on
your data looks wrong, the receipt makes it checkable, and that is exactly the
kind of issue we want. Run `pytest -q` before opening a PR.

## License

Apache-2.0.
