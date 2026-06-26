"""SignalMap wire-frame decoder (v1). Mirror of schema/frame.md.

Zero-copy-ish: we slice the buffer and use numpy views where possible.
The decoder is strict — it raises on bad magic/version so corruption is loud,
not silent.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC_RAW = 0x5247  # 'RG'
MAGIC_SPEC = 0x5347  # 'SG'
VERSION = 1
HEADER_LEN = 28

FLAG_SPECTRUM = 0b0000_0001
FLAG_PHASE = 0b0000_0010

# magic(H) version(B) flags(B) node_id(I) seq(I) ts_us(Q) sr_hz(I) n(H) reserved(H)
_HEADER = struct.Struct("<HBBIIQIHH")


@dataclass(frozen=True)
class Frame:
    is_spectrum: bool
    node_id: int
    seq: int
    ts_us: int
    sr_hz: int
    n: int
    payload: np.ndarray  # int16 (raw) or float32 (spectrum)
    sensor_class: int = 0  # METADATA ONLY — never fed to the model (bias-free)


def decode(buf: bytes) -> Frame:
    if len(buf) < HEADER_LEN:
        raise ValueError(f"frame too short: {len(buf)} < {HEADER_LEN}")
    magic, version, flags, node_id, seq, ts_us, sr_hz, n, sensor_class = _HEADER.unpack_from(buf, 0)
    if version != VERSION:
        raise ValueError(f"unsupported frame version {version}")
    if magic not in (MAGIC_RAW, MAGIC_SPEC):
        raise ValueError(f"bad magic {magic:#06x}")

    is_spectrum = magic == MAGIC_SPEC
    body = buf[HEADER_LEN:]
    if is_spectrum:
        payload = np.frombuffer(body, dtype="<f4", count=n)
    else:
        payload = np.frombuffer(body, dtype="<i2", count=n)
    return Frame(is_spectrum, node_id, seq, ts_us, sr_hz, n, payload, sensor_class & 0xFF)


def encode_raw(node_id: int, seq: int, ts_us: int, sr_hz: int, samples: np.ndarray,
               sensor_class: int = 0) -> bytes:
    """Encode a raw int16 block into a v1 frame — byte-identical to the ESP32
    firmware's frame::pack_raw. Shared by every node type (mic, salvaged
    transducer via ESP32, host USB capture). `sensor_class` rides in the
    reserved field as METADATA ONLY — the model never sees it (bias-free)."""
    s = np.asarray(samples, dtype="<i2")
    hdr = _HEADER.pack(MAGIC_RAW, VERSION, 0, node_id, seq, ts_us, sr_hz, len(s),
                       sensor_class & 0xFF)
    return hdr + s.tobytes()


class GapTracker:
    """Per-node seq continuity. A gap is DATA, never silently interpolated."""

    def __init__(self) -> None:
        self._last: dict[int, int] = {}

    def check(self, node_id: int, seq: int) -> int:
        """Return number of dropped frames since the last seen frame (0 = ok)."""
        last = self._last.get(node_id)
        self._last[node_id] = seq
        if last is None:
            return 0
        expected = (last + 1) & 0xFFFF_FFFF
        if seq == expected:
            return 0
        return (seq - expected) & 0xFFFF_FFFF
