#![no_std]

//! Allocation-free primitives shared by tiny SignalMap edge devices.
//! No networking, allocator, floating point, or platform HAL is required.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stats {
    pub count: u16,
    pub mean: i32,
    pub rms: u32,
    pub peak_abs: u32,
    pub zero_crossings: u16,
}

/// Compute bounded integer features in one pass. Empty input returns zeros.
pub fn stats(samples: &[i16]) -> Stats {
    if samples.is_empty() {
        return Stats {
            count: 0,
            mean: 0,
            rms: 0,
            peak_abs: 0,
            zero_crossings: 0,
        };
    }
    let mut sum = 0i64;
    let mut sum_sq = 0u64;
    let mut peak = 0u32;
    let mut crossings = 0u16;
    let mut previous = samples[0];
    for &sample in samples {
        let value = i64::from(sample);
        let magnitude = value.unsigned_abs() as u32;
        sum += value;
        sum_sq = sum_sq.saturating_add((value.unsigned_abs()).saturating_mul(value.unsigned_abs()));
        peak = peak.max(magnitude);
        if (previous < 0 && sample >= 0) || (previous >= 0 && sample < 0) {
            crossings = crossings.saturating_add(1);
        }
        previous = sample;
    }
    let count = samples.len().min(u16::MAX as usize) as u16;
    Stats {
        count,
        mean: (sum / samples.len() as i64) as i32,
        rms: isqrt(sum_sq / samples.len() as u64),
        peak_abs: peak,
        zero_crossings: crossings,
    }
}

fn isqrt(value: u64) -> u32 {
    let mut bit = 1u64 << 62;
    let mut result = 0u64;
    while bit > value {
        bit >>= 2;
    }
    let mut n = value;
    while bit != 0 {
        if n >= result + bit {
            n -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result.min(u32::MAX as u64) as u32
}

/// CRC-16/CCITT-FALSE, suitable for an optional frame integrity trailer.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc = 0xFFFFu16;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

pub mod frame {
    pub const MAGIC_RAW: u16 = 0x5247;
    pub const VERSION: u8 = 1;
    pub const HEADER_LEN: usize = 28;
    pub const MAX_SAMPLES: usize = 512;
    pub const MAX_FRAME: usize = HEADER_LEN + MAX_SAMPLES * 2;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum FrameError {
        OutputTooSmall,
        TooManySamples,
    }

    /// Pack the v1 raw frame without allocation or truncation.
    pub fn pack_raw(
        out: &mut [u8],
        node_id: u32,
        seq: u32,
        ts_us: u64,
        sr_hz: u32,
        samples: &[i16],
    ) -> Result<usize, FrameError> {
        if samples.len() > MAX_SAMPLES {
            return Err(FrameError::TooManySamples);
        }
        let needed = HEADER_LEN + samples.len() * 2;
        if out.len() < needed {
            return Err(FrameError::OutputTooSmall);
        }
        out[0..2].copy_from_slice(&MAGIC_RAW.to_le_bytes());
        out[2] = VERSION;
        out[3] = 0;
        out[4..8].copy_from_slice(&node_id.to_le_bytes());
        out[8..12].copy_from_slice(&seq.to_le_bytes());
        out[12..20].copy_from_slice(&ts_us.to_le_bytes());
        out[20..24].copy_from_slice(&sr_hz.to_le_bytes());
        out[24..26].copy_from_slice(&(samples.len() as u16).to_le_bytes());
        out[26..28].copy_from_slice(&0u16.to_le_bytes());
        for (i, sample) in samples.iter().enumerate() {
            let offset = HEADER_LEN + i * 2;
            out[offset..offset + 2].copy_from_slice(&sample.to_le_bytes());
        }
        Ok(needed)
    }
}
