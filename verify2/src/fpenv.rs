// CIS-2 v0.1 §1.3 FTZ/DAZ pinning + adversarial self-test. Implemented
// from spec text alone (E15g clean-room attempt #2).

#[cfg(target_arch = "x86_64")]
pub fn pin_ftz_daz() {
    unsafe {
        let mut csr: u32 = 0;
        std::arch::asm!("stmxcsr [{0}]", in(reg) &mut csr, options(nostack));
        csr |= 1 << 15; // FTZ
        csr |= 1 << 6; // DAZ
        std::arch::asm!("ldmxcsr [{0}]", in(reg) &csr, options(nostack));
        let mut readback: u32 = 0;
        std::arch::asm!("stmxcsr [{0}]", in(reg) &mut readback, options(nostack));
        assert!(readback & (1 << 15) != 0, "MXCSR FTZ bit not set on readback");
        assert!(readback & (1 << 6) != 0, "MXCSR DAZ bit not set on readback");
    }
}

#[cfg(target_arch = "aarch64")]
pub fn pin_ftz_daz() {
    unsafe {
        let mut fpcr: u64;
        std::arch::asm!("mrs {0}, fpcr", out(reg) fpcr);
        fpcr |= 1 << 24; // FZ
        std::arch::asm!("msr fpcr, {0}", in(reg) fpcr);
        let mut readback: u64;
        std::arch::asm!("mrs {0}, fpcr", out(reg) readback);
        assert!(readback & (1 << 24) != 0, "FPCR FZ bit not set on readback");
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn pin_ftz_daz() {
    panic!("CIS-2 v0.1 §1.3: unsupported ISA, no defined FTZ/DAZ control");
}

/// §1.3 adversarial self-test: with FTZ/DAZ pinned, denormal inputs and
/// outputs must flush to +0.0, checked through an optimization barrier so
/// the compiler cannot fold the arithmetic away.
pub fn adversarial_selftest() {
    let min_pos = std::hint::black_box(f32::MIN_POSITIVE);
    let tiny = std::hint::black_box(1.0e-10_f32);
    let product = min_pos * tiny;
    assert_eq!(product, 0.0_f32, "denormal product did not flush to +0.0");
    assert!(product.is_sign_positive());

    let smallest_subnormal = std::hint::black_box(f32::from_bits(0x00000001));
    let zero = std::hint::black_box(0.0_f32);
    let sum = smallest_subnormal + zero;
    assert_eq!(sum, 0.0_f32, "denormal input did not flush to +0.0 in add");
}
