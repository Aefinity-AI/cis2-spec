//! CIS-2 §3.4 — pin FTZ (flush-to-zero) and DAZ (denormals-are-zero) on at
//! process start, so this ISA's default doesn't silently diverge from the
//! other's (memo §2 "Denormals / FTZ / DAZ" row) — a real, hardware-level
//! cross-ISA divergence risk if left unpinned.
//!
//! x86_64: set MXCSR bits 15 (FTZ) and bit 6 (DAZ) directly (`_mm_setcsr`).
//!
//! aarch64 (E15c): the FPCR (Floating-point Control Register) FZ bit
//! (bit 24) controls flush-to-zero for both inputs and outputs of scalar
//! and Advanced SIMD binary32/binary64 arithmetic — unlike x86_64, aarch64
//! has no separate DAZ control; FZ alone flushes both denormal inputs and
//! denormal outputs. `aarch64`'s FPCR.FZ defaults to 0 (IEEE-754 gradual
//! underflow) on Linux/AAPCS64 unless explicitly set, so we set it via
//! inline asm (`mrs`/`msr fpcr`) at process start and assert the readback,
//! mirroring the x86_64 belt-and-suspenders check. FZ16 (bit 19, the
//! half-precision flush-to-zero control) is deliberately left untouched —
//! this reference never uses fp16.

/// Set FTZ=on, DAZ=on on x86_64. Must be called once at process start,
/// before any fp32 arithmetic in the decode path.
#[cfg(target_arch = "x86_64")]
pub fn pin_ftz_daz() {
    // Set MXCSR bits directly: bit 15 = FTZ, bit 6 = DAZ. Using
    // `_mm_setcsr`/`_mm_getcsr` (not the higher-level `_MM_SET_*` wrappers,
    // which are deprecated and DAZ has no safe wrapper in this toolchain's
    // core::arch surface) so the exact bits set are explicit and auditable.
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    const FTZ_BIT: u32 = 1 << 15;
    const DAZ_BIT: u32 = 1 << 6;
    unsafe {
        let csr = _mm_getcsr();
        _mm_setcsr(csr | FTZ_BIT | DAZ_BIT);
    }
}

/// Set FTZ (Flush-to-Zero) on aarch64, E15c. FPCR bit 24 (FZ) controls
/// flush-to-zero for scalar and Advanced SIMD binary32/binary64 arithmetic.
/// We deliberately do NOT set bit 19 (FZ16, the half-precision flush-to-zero
/// control) — this reference never uses fp16, and leaving it at its
/// architectural default keeps the pinned bit set minimal and auditable
/// (matches the module doc's original note: "FZ16 irrelevant, leave").
#[cfg(target_arch = "aarch64")]
pub fn pin_ftz_daz() {
    const FZ_BIT: u64 = 1 << 24;
    unsafe {
        let mut fpcr: u64;
        std::arch::asm!("mrs {0}, fpcr", out(reg) fpcr);
        fpcr |= FZ_BIT;
        std::arch::asm!("msr fpcr, {0}", in(reg) fpcr);
        let mut readback: u64;
        std::arch::asm!("mrs {0}, fpcr", out(reg) readback);
        assert!(
            readback & FZ_BIT != 0,
            "FPCR.FZ (bit 24) did not read back as set after msr fpcr"
        );
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn pin_ftz_daz() {
    compile_error!(
        "CIS-2 E15b/E15c have only pinned the x86_64 MXCSR and aarch64 FPCR.FZ paths — see module docs. Do not silently no-op on other ISAs."
    );
}

/// Read back MXCSR to confirm FTZ/DAZ are actually set (belt-and-suspenders
/// verification, not just "we called the setter").
#[cfg(target_arch = "x86_64")]
pub fn mxcsr_ftz_daz_on() -> bool {
    use std::arch::x86_64::_mm_getcsr;
    let csr = unsafe { _mm_getcsr() };
    let ftz = (csr & (1 << 15)) != 0;
    let daz = (csr & (1 << 6)) != 0;
    ftz && daz
}

/// Read back FPCR.FZ (bit 24) to confirm flush-to-zero is actually set.
#[cfg(target_arch = "aarch64")]
pub fn fpcr_fz_on() -> bool {
    const FZ_BIT: u64 = 1 << 24;
    let fpcr: u64;
    unsafe {
        std::arch::asm!("mrs {0}, fpcr", out(reg) fpcr);
    }
    fpcr & FZ_BIT != 0
}

/// Portable readback: true iff this ISA's flush-to-zero/denormals-are-zero
/// control bit(s) are pinned on. Used by `main()` and by the cross-ISA test
/// below so callers don't need per-arch `cfg` at the call site.
#[cfg(target_arch = "x86_64")]
pub fn ftz_daz_on() -> bool {
    mxcsr_ftz_daz_on()
}

#[cfg(target_arch = "aarch64")]
pub fn ftz_daz_on() -> bool {
    fpcr_fz_on()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftz_daz_pinned_and_denormal_input_flushed() {
        pin_ftz_daz();
        assert!(ftz_daz_on(), "FTZ/DAZ (or FPCR.FZ) must read back as ON");

        // Adversarial denormal test (§3.4): smallest positive normal times a
        // tiny value, which would underflow to a subnormal under gradual
        // underflow, must instead flush deterministically to +0.0 with
        // DAZ/FTZ pinned.
        let smallest_normal = f32::MIN_POSITIVE; // 2^-126
        let tiny = 1.0e-10f32;
        let product = smallest_normal * tiny;
        assert_eq!(
            product,
            0.0f32,
            "expected flush-to-zero, got {product:e} (bits {:#x})",
            product.to_bits()
        );
        assert!(product.is_sign_positive());

        // A genuinely subnormal literal, added to zero, must also flush.
        // `black_box` prevents LLVM from proving/optimizing away the add
        // (e.g. folding `x + 0.0` to `x`) so the hardware op actually runs.
        let denorm_bits: u32 = 0x0000_0001; // smallest positive subnormal
        let denorm = std::hint::black_box(f32::from_bits(denorm_bits));
        let flushed = std::hint::black_box(denorm) + std::hint::black_box(0.0f32);
        assert_eq!(
            flushed,
            0.0f32,
            "subnormal + 0.0 must flush to +0.0 under DAZ/FTZ (got bits {:#x})",
            flushed.to_bits()
        );
    }
}
