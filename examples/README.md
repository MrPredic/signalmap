# Examples — real use cases, same two commands

Every example follows the identical pattern — **`fit` on healthy data, `monitor`
for anomalies** — across different sensor types. That sameness *is* the platform:
one workflow, any modality.

```
signalmap fit     --dataset <healthy>.parquet --healthy-label normal --out det.pt
signalmap monitor --source replay --dataset <stream>.parquet --detector det.pt
```

## ✅ Vibration — real CWRU bearings (reproducible)
Downloads real Case Western Reserve bearing data and reproduces the headline
result, end to end:
```bash
pip install scipy pyarrow            # one-time
python examples/fetch_cwru.py        # -> data/cwru_real.parquet
signalmap fit     --dataset data/cwru_real.parquet --healthy-label normal --out det.pt --epochs 40
signalmap monitor --source replay --dataset data/cwru_real.parquet --detector det.pt --quiet
```
Expected: **238/238 faults caught, ~0.2% false alarms**, fully unsupervised.

## 🔊 Acoustic — machine sounds (recipe; bring your own audio)
Any WAV works — record a healthy machine, then a faulty one (or use the MIMII
dataset, zenodo.org/records/3384388):
```bash
signalmap ingest-file healthy_machine.wav --label normal          --out data/audio.parquet
signalmap ingest-file faulty_machine.wav  --label ANOMALY_fault   --out data/audio.parquet
signalmap fit     --dataset data/audio.parquet --healthy-label normal --out det.pt
signalmap monitor --source replay --dataset data/audio.parquet --detector det.pt
```

## ⚡ Electrical / current — CSV logs (recipe)
A current/voltage log from any cheap sensor (one value per row):
```bash
signalmap ingest-file motor_ok.csv    --label normal        --column 0 --sr 10000 --out data/elec.parquet
signalmap ingest-file motor_fault.csv --label ANOMALY_fault --column 0 --sr 10000 --out data/elec.parquet
signalmap fit     --dataset data/elec.parquet --healthy-label normal --out det.pt
signalmap monitor --source replay --dataset data/elec.parquet --detector det.pt
```

## 🧪 No hardware? Pure synthetic
```bash
signalmap benchmark      # ROC-AUC on a synthetic predictive-maintenance set
signalmap universal      # 8 sensor domains, one model, cross-domain proof
signalmap discover       # cross-modal coupling vs. confound
```

> Honesty note: a monitor alert flags a deviation from healthy baseline. On clean
> benchmarks (CWRU) that is near-perfect; harder real-world cases are ongoing
> work. An alert is a signal to inspect, not a certified diagnosis.
