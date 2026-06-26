//! SignalMap edge firmware — raw ADC streaming (ESP32-S3, no_std).
//!
//! Design contract (the whole point of the project):
//!   * NO software filtering, NO AGC, NO DC removal, NO scaling on the edge.
//!   * Raw int16 ADC samples are packed verbatim into a versioned binary frame.
//!   * `seq` is monotonic so the backend can detect (never hide) sample loss.
//!
//! Two parallel paths (see schema/frame.md):
//!   * RAW       (magic 'RG') — always emitted, the bias-free ground truth.
//!   * SPECTRUM  (magic 'SG') — optional, edge-side FFT magnitude to save
//!     bandwidth when the link is constrained. Off by default.
//!
//! This file is a reference scaffold: ADC sampling and frame packing are
//! complete; Wi-Fi/MQTT bring-up is sketched with clear board-specific markers
//! (`>>>`). Swap the oneshot loop for I2S-DMA in production for jitter-free HF.

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    analog::adc::{Adc, AdcConfig, Attenuation},
    clock::CpuClock,
    gpio::Io,
    timer::timg::TimerGroup,
};
use esp_println::println;

// ---------------------------------------------------------------------------
// Wire frame (v1) — see schema/frame.md. Header is 28 bytes, little-endian.
// ---------------------------------------------------------------------------
pub mod frame {
    pub const MAGIC_RAW: u16 = 0x5247; // 'RG'
    pub const MAGIC_SPEC: u16 = 0x5347; // 'SG'
    pub const VERSION: u8 = 1;
    pub const HEADER_LEN: usize = 28;

    pub const FLAG_SPECTRUM: u8 = 0b0000_0001;
    pub const FLAG_PHASE: u8 = 0b0000_0010;

    /// Max raw samples per frame. Tune against MQTT MTU / sample rate.
    pub const MAX_SAMPLES: usize = 512;
    pub const MAX_FRAME: usize = HEADER_LEN + MAX_SAMPLES * 2;

    /// Pack a raw int16 block into `out`, returns bytes written.
    pub fn pack_raw(
        out: &mut [u8; MAX_FRAME],
        node_id: u32,
        seq: u32,
        ts_us: u64,
        sr_hz: u32,
        samples: &[i16],
    ) -> usize {
        let n = samples.len().min(MAX_SAMPLES);
        out[0..2].copy_from_slice(&MAGIC_RAW.to_le_bytes());
        out[2] = VERSION;
        out[3] = 0; // flags: raw, no phase
        out[4..8].copy_from_slice(&node_id.to_le_bytes());
        out[8..12].copy_from_slice(&seq.to_le_bytes());
        out[12..20].copy_from_slice(&ts_us.to_le_bytes());
        out[20..24].copy_from_slice(&sr_hz.to_le_bytes());
        out[24..26].copy_from_slice(&(n as u16).to_le_bytes());
        out[26..28].copy_from_slice(&0u16.to_le_bytes()); // reserved
        for (i, s) in samples[..n].iter().enumerate() {
            let off = HEADER_LEN + i * 2;
            out[off..off + 2].copy_from_slice(&s.to_le_bytes());
        }
        HEADER_LEN + n * 2
    }
}

const NODE_ID: u32 = 0x0000_0001; // >>> assign unique per device (e.g. from eFuse MAC)
const SAMPLE_RATE_HZ: u32 = 16_000; // >>> match your I2S-DMA config in production

#[esp_hal::main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let io = Io::new(peripherals.IO_MUX);
    let _timg0 = TimerGroup::new(peripherals.TIMG0);

    // --- ADC setup: GPIO1 / ADC1, 11dB attenuation = full ~0..3.3V range. ---
    // No attenuation trickery for "nice" ranges — we want the unbiased signal.
    let mut adc_cfg = AdcConfig::new();
    let mut pin = adc_cfg.enable_pin(io.pins.gpio1, Attenuation::Attenuation11dB);
    let mut adc = Adc::new(peripherals.ADC1, adc_cfg);

    let mut seq: u32 = 0;
    let mut buf = [0u8; frame::MAX_FRAME];
    let mut samples = [0i16; frame::MAX_SAMPLES];

    println!("signalmap-fw: node={NODE_ID:#010x} sr={SAMPLE_RATE_HZ}Hz");

    loop {
        // ----- Acquire one raw block -----------------------------------------
        // PRODUCTION: replace this oneshot loop with I2S-DMA into a lock-free
        // ring buffer for jitter-free high-frequency sampling. Oneshot here
        // keeps the reference dependency-light and easy to read.
        let ts_us = embassy_time::Instant::now().as_micros();
        for s in samples.iter_mut() {
            // 12-bit unsigned (0..4095) -> centered int16, raw, no smoothing.
            let raw: u16 = nb_block(|| adc.read_oneshot(&mut pin));
            *s = (raw as i32 - 2048) as i16;
        }

        // ----- Pack & publish -------------------------------------------------
        let len = frame::pack_raw(&mut buf, NODE_ID, seq, ts_us, SAMPLE_RATE_HZ, &samples);
        publish(b"signals/1/raw", &buf[..len]);
        seq = seq.wrapping_add(1);

        // OPTIONAL parallel spectrum path (off by default):
        //   use microfft::real::rfft_512 on a f32 copy of `samples`,
        //   take magnitudes, pack with frame::MAGIC_SPEC, publish to
        //   "signals/1/spectrum". Raw remains the source of truth.
    }
}

/// Minimal blocking helper for `nb::Result` ADC reads.
fn nb_block<T, E>(mut f: impl FnMut() -> nb::Result<T, E>) -> T {
    loop {
        if let Ok(v) = f() {
            return v;
        }
    }
}

/// Network publish abstraction.
///
/// >>> Wire this to an embassy task that owns the Wi-Fi stack + rust-mqtt
/// client. Kept as a stub so the sampling/packing core is testable in
/// isolation and the frame format is the documented contract. Recommended:
///   * esp-wifi -> embassy-net stack (DHCP)
///   * rust-mqtt MqttClient, QoS1, topic "signals/{node_id}/raw"
///   * back-pressure: drop oldest frame if the TX queue is full (never block
///     the sampler) and increment a dropped-frame counter in the next header's
///     reserved field.
fn publish(_topic: &[u8], _payload: &[u8]) {
    // no-op in the scaffold; replace with MQTT send.
}
