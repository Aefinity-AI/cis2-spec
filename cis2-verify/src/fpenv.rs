//! Spec 1.3: the floating-point environment MUST be pinned before any
//! decode-path arithmetic, and the pin MUST be verified by readback plus an
//! adversarial self-test — "we called the setter" is explicitly not enough.
//!
//! Spec 1.3 also says any ISA other than x86-64 and aarch64 is undefined by
//! v0.3b and a conforming implementation MUST refuse to run rather than
//! silently proceed. That refusal is a compile error here, not a runtime
//! branch, so a non-conforming target cannot be built at all.

use core::hint::black_box;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpEnvError {
    /// The control register was written but did not read back with the
    /// required bits set.
    ReadbackFailed,
    /// `MIN_POSITIVE * 1e-10` did not flush to exactly `+0.0`.
    UnderflowNotFlushed,
    /// `f32::from_bits(1) + 0.0` did not flush to exactly `+0.0`.
    DenormalInputNotZeroed,
}

#[cfg(target_arch = "x86_64")]
mod arch {
    /// Read MXCSR directly with `stmxcsr` (spec 1.3: a direct
    /// read-modify-write of the register, not the deprecated `_MM_SET_*`
    /// wrapper macros).
    #[inline]
    pub fn read_control() -> u32 {
        let mut v: u32 = 0;
        unsafe {
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut v, options(nostack, preserves_flags));
        }
        v
    }

    #[inline]
    pub fn write_control(v: u32) {
        unsafe {
            core::arch::asm!("ldmxcsr [{}]", in(reg) &v, options(nostack, preserves_flags));
        }
    }

    /// MXCSR bit 15 = FTZ (flush-to-zero), bit 6 = DAZ (denormals-are-zero).
    pub const REQUIRED: u32 = (1 << 15) | (1 << 6);
    pub const NAME: &str = "x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)";
}

#[cfg(target_arch = "aarch64")]
mod arch {
    /// Read FPCR directly with `mrs` (spec 1.3).
    #[inline]
    pub fn read_control() -> u64 {
        let v: u64;
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) v, options(nomem, nostack, preserves_flags));
        }
        v
    }

    #[inline]
    pub fn write_control(v: u64) {
        unsafe {
            core::arch::asm!("msr fpcr, {}", in(reg) v, options(nomem, nostack, preserves_flags));
        }
    }

    /// FPCR bit 24 = FZ. aarch64 has no separate DAZ control; FZ alone
    /// flushes both denormal inputs and denormal outputs. Bit 19 (FZ16) is
    /// deliberately left at its architectural default — spec 1.3 says do
    /// not set it.
    pub const REQUIRED: u64 = 1 << 24;
    pub const NAME: &str = "aarch64 FPCR FZ(bit 24)";
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!(
    "CIS-2 v0.3b section 1.3 leaves the FTZ/DAZ control undefined on this ISA and requires a \
     conforming implementation to refuse to run rather than silently proceed without an \
     equivalent control. Only x86_64 and aarch64 are conformant targets for this version."
);

/// Name of the control bits this build pins, for the receipt/report.
pub fn control_name() -> &'static str {
    arch::NAME
}

/// Pin FTZ/DAZ, assert by readback, then run spec 1.3's adversarial
/// self-test. Every operand goes through `black_box` so the compiler cannot
/// constant-fold the arithmetic and mask a broken pin.
pub fn pin_and_selftest() -> Result<(), FpEnvError> {
    let cur = arch::read_control();
    arch::write_control(cur | arch::REQUIRED);

    // Readback, not "we called the setter".
    let back = arch::read_control();
    if back & arch::REQUIRED != arch::REQUIRED {
        return Err(FpEnvError::ReadbackFailed);
    }

    // Denormal OUTPUT must flush: 2^-126 * 1e-10 is subnormal without FTZ.
    let tiny = black_box(f32::MIN_POSITIVE);
    let scale = black_box(1.0e-10f32);
    let prod = black_box(tiny * scale);
    if prod.to_bits() != 0 {
        return Err(FpEnvError::UnderflowNotFlushed);
    }

    // Denormal INPUT must be treated as zero: the smallest positive
    // subnormal added to +0.0 is itself without DAZ/FZ.
    let sub = black_box(f32::from_bits(0x0000_0001));
    let zero = black_box(0.0f32);
    let sum = black_box(sub + zero);
    if sum.to_bits() != 0 {
        return Err(FpEnvError::DenormalInputNotZeroed);
    }

    Ok(())
}

/// Re-assert the readback without re-writing, for callers that want to
/// check the environment has not been disturbed mid-run.
pub fn is_pinned() -> bool {
    arch::read_control() & arch::REQUIRED == arch::REQUIRED
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operand goes through `black_box` in both directions. Without
    /// that, LLVM folds `f32::from_bits(K1) * f32::from_bits(K2)` at compile
    /// time under the *host compiler's* rounding rules, which do not honour a
    /// runtime MXCSR/FPCR pin — the test then reports the compiler's answer
    /// and passes whether or not the pin works. This was observed: the first
    /// draft of these goldens printed unflushed subnormals on a demonstrably
    /// pinned process.
    fn mul(ab: u32, bb: u32) -> u32 {
        let a = black_box(f32::from_bits(black_box(ab)));
        let b = black_box(f32::from_bits(black_box(bb)));
        black_box(a * b).to_bits()
    }
    fn add(ab: u32, bb: u32) -> u32 {
        let a = black_box(f32::from_bits(black_box(ab)));
        let b = black_box(f32::from_bits(black_box(bb)));
        black_box(a + b).to_bits()
    }

    /// (a, b, op, pinned result) — the bits a conforming implementation MUST
    /// produce under 1.3, on x86-64 and on aarch64 alike.
    const FTZ: &[(u32, u32, char, u32)] = &[
        // Products whose exact value is subnormal: FTZ must flush them.
        (0x2000_0000, 0x1F80_0000, '*', 0x0000_0000), // 2^-63 * 2^-64  = 2^-127
        (0x3380_0000, 0x0C00_0000, '*', 0x0000_0000), // 2^-24 * 2^-103 = 2^-127
        // Cancellation of two normals into the subnormal range.
        (0x0080_0001, 0x8080_0000, '+', 0x0000_0000), // 2^-126(1+2^-23) - 2^-126
        // Subnormal INPUTS: DAZ/FZ must read them as zero.
        (0x0000_0001, 0x0000_0000, '+', 0x0000_0000), // 2^-149 + 0
        (0x0000_0001, 0x3F80_0000, '*', 0x0000_0000), // 2^-149 * 1
        (0x0000_0001, 0x7F00_0000, '*', 0x0000_0000), // 2^-149 * 2^127
    ];

    /// Cases that come out identical pinned or not, kept so the discrimination
    /// test below cannot accidentally be satisfied by them.
    const INERT: &[(u32, u32, char, u32)] = &[
        (0x0C80_0000, 0x2000_0000, '*', 0x0000_0000), // 2^-165: below subnormal range
        (0x0000_0001, 0x8000_0001, '+', 0x0000_0000), // exact cancellation
    ];

    fn eval(a: u32, b: u32, op: char) -> u32 {
        if op == '*' { mul(a, b) } else { add(a, b) }
    }

    #[test]
    fn pinned_denormal_goldens() {
        pin_and_selftest().expect("1.3 pin");
        assert!(is_pinned());
        for &(a, b, op, want) in FTZ.iter().chain(INERT) {
            assert_eq!(
                eval(a, b, op), want,
                "0x{a:08X} {op} 0x{b:08X} under {}", control_name()
            );
        }
    }

    /// The goldens are only worth publishing if an implementation that ignores
    /// 1.3 gets different bits. Clearing the pin must move every FTZ case and
    /// no INERT case. Runs in the harness's own thread, and the control word is
    /// per-thread on both supported ISAs, so this cannot leak into other tests.
    #[test]
    fn clearing_the_pin_changes_the_answers() {
        pin_and_selftest().expect("1.3 pin");
        let saved = arch::read_control();
        arch::write_control(saved & !arch::REQUIRED);
        assert!(!is_pinned(), "could not clear the pin, so this proves nothing");

        let unpinned: Vec<u32> = FTZ.iter().map(|&(a, b, op, _)| eval(a, b, op)).collect();
        let inert: Vec<u32> = INERT.iter().map(|&(a, b, op, _)| eval(a, b, op)).collect();

        arch::write_control(saved);
        assert!(is_pinned(), "failed to restore the pin");

        for (i, &(a, b, op, want)) in FTZ.iter().enumerate() {
            assert_ne!(
                unpinned[i], want,
                "0x{a:08X} {op} 0x{b:08X} gives the same bits pinned and unpinned, \
                 so it does not test 1.3 at all"
            );
        }
        for (i, &(_, _, _, want)) in INERT.iter().enumerate() {
            assert_eq!(inert[i], want, "inert case moved; the table is mislabelled");
        }
    }
}
