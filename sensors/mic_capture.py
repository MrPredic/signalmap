"""Laptop microphone = the first SignalMap node (free 44.1 kHz ADC).

The mic is treated exactly like an ESP32: raw int16 samples are sliced into
v1 wire frames (frame.encode_raw). No filtering — the OS mic path is the only
unavoidable bias, and we document it rather than add more.

Modes
-----
record:  capture N seconds, slice into frames, append to a Parquet dataset for
         training/replay. `--label` is *session metadata only* — it is NOT fed
         to the model (the project stays norm/label free); we keep it solely so
         humans can later check whether clusters separate.

    python -m sensors.mic_capture record --label glass_tap --seconds 8 \
        --out data/dataset.parquet

stream:  publish frames live to MQTT (acts as a real ESP32 node).

    python -m sensors.mic_capture stream --broker localhost --node 1001

list:    show input devices.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from signalmap.frame import encode_raw

FRAME_N = 512  # samples per frame, matches firmware MAX_SAMPLES default


def _record_block(seconds: float, sr: int) -> np.ndarray:
    import sounddevice as sd

    print(f"  recording {seconds}s @ {sr} Hz ... (make the sound now)")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    # float32 [-1,1] -> centered int16-range counts, raw (no AGC, no filter).
    return np.clip(audio[:, 0] * 2047.0, -2048, 2047).astype(np.int16)


def _to_frames(samples: np.ndarray) -> list[np.ndarray]:
    n_full = len(samples) // FRAME_N
    return [samples[i * FRAME_N:(i + 1) * FRAME_N] for i in range(n_full)]


def cmd_record(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    samples = _record_block(args.seconds, args.sr)
    frames = _to_frames(samples)
    print(f"  {len(frames)} frames captured")

    rows = {
        "label": [args.label] * len(frames),
        "sr_hz": [args.sr] * len(frames),
        "ts_us": [int(time.time() * 1e6) + i for i in range(len(frames))],
        "samples": [f.astype("<i2").tobytes() for f in frames],
    }
    table = pa.table(rows)

    import os

    if os.path.exists(args.out):
        existing = pq.read_table(args.out)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, args.out)
    print(f"  appended -> {args.out} (total rows: {table.num_rows})")


def cmd_stream(args: argparse.Namespace) -> None:
    import sounddevice as sd

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise SystemExit("stream mode needs paho-mqtt: pip install paho-mqtt")

    client = mqtt.Client()
    client.connect(args.broker, 1883, 60)
    client.loop_start()
    topic = f"signals/{args.node}/raw"
    seq = 0
    print(f"  streaming mic -> {args.broker} topic={topic} (Ctrl-C to stop)")

    def callback(indata, frames_count, time_info, status):
        nonlocal seq
        block = np.clip(indata[:, 0] * 2047.0, -2048, 2047).astype(np.int16)
        for f in _to_frames(block):
            payload = encode_raw(args.node, seq, int(time.time() * 1e6), args.sr, f)
            client.publish(topic, payload, qos=1)
            seq += 1

    with sd.InputStream(samplerate=args.sr, channels=1, dtype="float32",
                        blocksize=FRAME_N * 8, callback=callback):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n  stopped after {seq} frames")
    client.loop_stop()


def cmd_list(_args: argparse.Namespace) -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main() -> None:
    p = argparse.ArgumentParser(description="SignalMap mic node")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--label", required=True, help="session metadata only (not used by model)")
    r.add_argument("--seconds", type=float, default=8.0)
    r.add_argument("--sr", type=int, default=44100)
    r.add_argument("--out", default="data/dataset.parquet")
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("stream")
    s.add_argument("--broker", default="localhost")
    s.add_argument("--node", type=int, default=1001)
    s.add_argument("--sr", type=int, default=44100)
    s.set_defaults(func=cmd_stream)

    sub.add_parser("list").set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
