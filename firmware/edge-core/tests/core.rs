use signalmap_edge_core::frame::{pack_raw, FrameError, HEADER_LEN, MAX_SAMPLES};
use signalmap_edge_core::{crc16_ccitt, stats, Stats};

#[test]
fn stats_are_integer_golden_vector_and_bounded() {
    let samples = [-4i16, -2, 0, 2, 4, 2, -2];
    let got = stats(&samples);
    assert_eq!(
        got,
        Stats {
            count: 7,
            mean: 0,
            rms: 2,
            peak_abs: 4,
            zero_crossings: 2,
        }
    );
}

#[test]
fn stats_handle_extreme_i16_without_overflow() {
    let samples = [i16::MIN, i16::MAX, i16::MIN, i16::MAX];
    let got = stats(&samples);
    assert_eq!(got.count, 4);
    assert_eq!(got.peak_abs, 32768);
    assert!(got.rms > 32000);
}

#[test]
fn crc_matches_known_vector() {
    assert_eq!(crc16_ccitt(b"123456789"), 0x29B1);
}

#[test]
fn raw_frame_matches_wire_contract() {
    let mut out = [0u8; HEADER_LEN + 3 * 2];
    let n = pack_raw(&mut out, 7, 11, 13, 16000, &[1, -2, 3]).unwrap();
    assert_eq!(n, 34);
    assert_eq!(&out[0..2], &0x5247u16.to_le_bytes());
    assert_eq!(out[2], 1);
    assert_eq!(&out[4..8], &7u32.to_le_bytes());
    assert_eq!(&out[8..12], &11u32.to_le_bytes());
    assert_eq!(&out[20..24], &16000u32.to_le_bytes());
    assert_eq!(&out[24..26], &3u16.to_le_bytes());
    assert_eq!(&out[28..34], &[1, 0, 254, 255, 3, 0]);
}

#[test]
fn frame_packing_fails_closed_on_small_output_or_oversize_input() {
    let mut short = [0u8; HEADER_LEN - 1];
    assert_eq!(
        pack_raw(&mut short, 0, 0, 0, 0, &[1]),
        Err(FrameError::OutputTooSmall)
    );
    let samples = [0i16; MAX_SAMPLES + 1];
    let mut out = [0u8; HEADER_LEN + MAX_SAMPLES * 2 + 1];
    assert_eq!(
        pack_raw(&mut out, 0, 0, 0, 0, &samples),
        Err(FrameError::TooManySamples)
    );
}
