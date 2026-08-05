# SignalMap Edge Core

Allocation-free `no_std` primitives for the smallest supported devices.

This crate is intentionally independent of Wi-Fi, MQTT, HALs, NumPy, and ML
runtimes. It provides:

- one-pass integer signal statistics;
- CRC-16/CCITT for an optional integrity trailer;
- strict v1 raw-frame packing with no truncation;
- fixed-size buffers only.

It is additive to `firmware/`: the existing ESP32-S3 raw-streaming behavior is
unchanged by default. The firmware opts into the strict packer with
`--features edge-core`; this does not enable any new network or ML dependency.
The release `.rlib` size is a diagnostic only; final device flash/RAM must be
measured on the target with the linker map and worst-case stack analysis.

Checks:

```bash
cargo test --manifest-path firmware/edge-core/Cargo.toml
cargo build --manifest-path firmware/edge-core/Cargo.toml --release

# target build, once the ESP Xtensa toolchain/target is installed:
cargo build --manifest-path firmware/Cargo.toml --release --features edge-core
```
