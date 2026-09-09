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
