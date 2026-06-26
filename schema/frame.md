# SignalMap Wire Frame (v1)

Custom packed binary frame. **Bias-free by design**: payload carries raw ADC
samples with *no* edge-side filtering, AGC, DC-removal or scaling.

All integers little-endian. Header is 28 bytes, payload follows immediately.

```
Offset  Field      Type      Notes
0       magic      u16       0x5247 ('RG') = raw, 0x5347 ('SG') = spectrum
2       version    u8        = 1
3       flags      u8        bit0: spectrum payload, bit1: phase channel present
4       node_id    u32       stable per-device id
8       seq        u32       monotonic, gap = sample loss (never silently filled)
12      ts_us      u64       device monotonic timestamp, microseconds
20      sr_hz      u32       sample rate in Hz (raw) / original sr for spectrum
24      n          u16       element count in payload
26      reserved   u16       = 0 (future: gain index, channel id)
28      payload    [..]      raw: int16[n]   spectrum: f32[n] magnitude
```

## Why custom binary (and why it stays)
- Edge: zero dependencies on `no_std`, deterministic, maximal throughput for
  high-frequency `int16` streams.
- The `version` byte makes migration to FlatBuffers/Protobuf an additive,
  non-breaking change if schema evolution is ever needed.

## Integrity rule
The decoder MUST track `seq` per `node_id` and emit a `gap` metric on
discontinuity. We never interpolate missing samples — a gap is data, not noise.
