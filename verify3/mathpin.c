/* mathpin.c — CIS-2 v0.2 pinned fp32 primitives (spec §1, §5, §6). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathpin.h"

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

/* ---- §1.3 FTZ/DAZ ---- */

void cis2_pin_ftz_daz(void)
{
#if defined(__x86_64__) || defined(__i386__)
    unsigned int csr = _mm_getcsr();
    csr |= (1u << 15); /* FTZ */
    csr |= (1u << 6);  /* DAZ */
    _mm_setcsr(csr);
    unsigned int readback = _mm_getcsr();
    if (!(readback & (1u << 15)) || !(readback & (1u << 6))) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: MXCSR FTZ/DAZ pin failed readback (0x%x)\n", readback);
        exit(97);
    }
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24); /* FZ */
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
    uint64_t readback;
    __asm__ volatile("mrs %0, fpcr" : "=r"(readback));
    if (!(readback & (1ull << 24))) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: FPCR.FZ pin failed readback (0x%llx)\n",
                (unsigned long long)readback);
        exit(97);
    }
#else
    fprintf(stderr, "CIS2_VERIFY3 FATAL: unsupported ISA for FTZ/DAZ pin (spec §1.3)\n");
    exit(97);
#endif
}

static float cis2_barrier_f32(float x)
{
    /* optimization barrier equivalent to std::hint::black_box for our purposes */
    volatile float v = x;
    return v;
}

void cis2_ftz_daz_selftest(void)
{
    float min_pos = cis2_f32_from_bits(0x00800000u); /* 2^-126 */
    float r1 = cis2_barrier_f32(min_pos) * cis2_barrier_f32(1.0e-10f);
    if (cis2_f32_bits(r1) != 0x00000000u) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: FTZ self-test 1 failed (got bits 0x%08x)\n", cis2_f32_bits(r1));
        exit(98);
    }
    float smallest_sub = cis2_f32_from_bits(0x00000001u);
    float r2 = cis2_barrier_f32(smallest_sub) + cis2_barrier_f32(0.0f);
    if (cis2_f32_bits(r2) != 0x00000000u) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: FTZ self-test 2 failed (got bits 0x%08x)\n", cis2_f32_bits(r2));
        exit(98);
    }
}

/* ---- §6.1 rsqrt ---- */
float cis2_rsqrt(float x)
{
    return 1.0f / sqrtf(x);
}

/* ---- §6.2 exp ---- */
#define EXP_LOG2E_BITS 0x3FB8AA3Bu
#define EXP_C1_BITS    0x3F318000u
#define EXP_C2_BITS    0xB95E8083u
static const uint32_t EXP_P_BITS[6] = {
    0x39506967u, 0x3AB743CEu, 0x3C088908u, 0x3D2AA9C1u, 0x3E2AAAAAu, 0x3F000000u
};

static float ldexp_exact(float x, int k)
{
    if (x == 0.0f) return x;
    uint32_t bits = cis2_f32_bits(x);
    int32_t exp_bits = (int32_t)((bits >> 23) & 0xFFu);
    int32_t new_exp = exp_bits + k;
    if (new_exp <= 0) return 0.0f;
    if (new_exp >= 0xFF) {
        int neg = (bits >> 31) & 1u;
        return neg ? -INFINITY : INFINITY;
    }
    uint32_t new_bits = (bits & ~((uint32_t)0xFFu << 23)) | ((uint32_t)new_exp << 23);
    return cis2_f32_from_bits(new_bits);
}

float cis2_exp_pinned(float x)
{
    if (isnan(x)) return NAN;
    if (x > 88.0f) return INFINITY;
    if (x < -88.0f) return 0.0f;

    float EXP_LOG2E = cis2_f32_from_bits(EXP_LOG2E_BITS);
    float EXP_C1 = cis2_f32_from_bits(EXP_C1_BITS);
    float EXP_C2 = cis2_f32_from_bits(EXP_C2_BITS);

    float t = x * EXP_LOG2E;
    float z0 = t + 0.5f;
    float k = floorf(z0);
    float kc1 = k * EXP_C1;
    float r0 = x - kc1;
    float kc2 = k * EXP_C2;
    float r = r0 - kc2;

    float r2 = r * r;
    float EXP_P0 = cis2_f32_from_bits(EXP_P_BITS[0]);
    float poly = EXP_P0;
    for (int i = 1; i <= 5; i++) {
        poly = poly * r + cis2_f32_from_bits(EXP_P_BITS[i]);
    }
    float m1 = poly * r2;
    float poly2 = m1 + r;
    float result = poly2 + 1.0f;

    return ldexp_exact(result, (int)k);
}

/* ---- §6.3 sin/cos ---- */
#define TWO_PI_HI_BITS 0x40C90FDAu
#define TWO_PI_LO_BITS 0x347FF14Du

static const uint32_t SIN_COEF_BITS[11] = {
    0x3F800000u, 0xBE2AAAABu, 0x3C088888u, 0xB9500D01u,
    0x3638EF1Eu, 0xB2D7322Cu, 0x2F30922Eu, 0xAB573FA0u,
    0x274A963Au, 0xA317A4DAu, 0x1EB8DC77u
};
static const uint32_t COS_COEF_BITS[11] = {
    0x3F800000u, 0xBF000000u, 0x3D2AAAACu, 0xBAB60B62u,
    0x37D00D02u, 0xB493F27Eu, 0x310F76C9u, 0xAD49CBAAu,
    0x29573F9Eu, 0xA53413C5u, 0x20F2A15Fu
};

static void two_pi_reduce(float x, float *r_out)
{
    float TWO_PI_HI = cis2_f32_from_bits(TWO_PI_HI_BITS);
    float TWO_PI_LO = cis2_f32_from_bits(TWO_PI_LO_BITS);
    float k = roundf(x / TWO_PI_HI);
    float khi = k * TWO_PI_HI;
    float r0 = x - khi;
    float klo = k * TWO_PI_LO;
    float r = r0 - klo;
    *r_out = r;
}

float cis2_cos_pinned(float x)
{
    float r;
    two_pi_reduce(x, &r);
    float r2 = r * r;
    float poly = cis2_f32_from_bits(COS_COEF_BITS[10]);
    for (int i = 9; i >= 0; i--) {
        poly = poly * r2 + cis2_f32_from_bits(COS_COEF_BITS[i]);
    }
    return poly;
}

float cis2_sin_pinned(float x)
{
    float r;
    two_pi_reduce(x, &r);
    float r2 = r * r;
    float poly = cis2_f32_from_bits(SIN_COEF_BITS[10]);
    for (int i = 9; i >= 0; i--) {
        poly = poly * r2 + cis2_f32_from_bits(SIN_COEF_BITS[i]);
    }
    return poly * r;
}

/* ---- §6.4 SiLU ---- */
float cis2_silu_pinned(float x)
{
    float neg = -x;
    float e = cis2_exp_pinned(neg);
    float denom = 1.0f + e;
    return x / denom;
}

/* ---- §6.5 ln ---- */
#define LOG_SQRTHF_BITS 0x3F3504F3u
#define LOG_Q1_BITS     0xB95E8083u
#define LOG_Q2_BITS     0x3F318000u
static const uint32_t LOG_P_BITS[9] = {
    0x3D9021BBu, 0xBDEBD1B8u, 0x3DEF251Bu, 0xBDFE5D4Fu, 0x3E11E9BFu,
    0xBE2AAE50u, 0x3E4CCEADu, 0xBE7FFFFCu, 0x3EAAAAAAu
};

static void frexp_exact(float x, float *mantissa_out, int32_t *exponent_out)
{
    uint32_t bits = cis2_f32_bits(x);
    int32_t exp_bits = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t mantissa_bits = (bits & 0x807FFFFFu) | ((uint32_t)126u << 23);
    *mantissa_out = cis2_f32_from_bits(mantissa_bits);
    *exponent_out = exp_bits - 126;
}

float cis2_ln_pinned(float x)
{
    if (isnan(x) || x < 0.0f) return NAN;
    if (x == 0.0f) return -INFINITY;

    float m; int32_t e;
    frexp_exact(x, &m, &e);

    float LOG_SQRTHF = cis2_f32_from_bits(LOG_SQRTHF_BITS);
    float LOG_Q1 = cis2_f32_from_bits(LOG_Q1_BITS);
    float LOG_Q2 = cis2_f32_from_bits(LOG_Q2_BITS);

    if (m < LOG_SQRTHF) {
        e = e - 1;
        m = m + m - 1.0f;
    } else {
        m = m - 1.0f;
    }

    float z = m * m;
    float poly = cis2_f32_from_bits(LOG_P_BITS[0]);
    for (int i = 1; i <= 8; i++) {
        poly = poly * m + cis2_f32_from_bits(LOG_P_BITS[i]);
    }
    float y = poly * m;
    y = y * z;
    float fe = (float)e;
    float t1 = fe * LOG_Q1;
    y = y + t1;
    float half_z = 0.5f * z;
    y = y - half_z;
    float result = m + y;
    float t2 = fe * LOG_Q2;
    result = result + t2;
    return result;
}

/* ---- §5.1 / §5.2 / §5.3 ---- */

float cis2_dot_seq(const float *a, const float *b, size_t n)
{
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float p = a[i] * b[i];
        acc = acc + p;
    }
    return acc;
}

float cis2_sum_seq(const float *a, size_t n)
{
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc = acc + a[i];
    }
    return acc;
}

void cis2_matvec(const float *w, const float *x, float *y, size_t out_features, size_t in_features)
{
    for (size_t o = 0; o < out_features; o++) {
        y[o] = cis2_dot_seq(w + o * in_features, x, in_features);
    }
}

/* ---- §8 RMSNorm ---- */
void cis2_rmsnorm(const float *x, const float *weight, float eps, float *out, size_t n)
{
    /* sq[i] then sum_seq — do it via a temp buffer to keep the spec's exact
     * left-to-right sum order (not fused into the main loop). */
    float *sq = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) sq[i] = x[i] * x[i];
    float ss = cis2_sum_seq(sq, n);
    free(sq);
    float mean = ss / (float)n;
    float inv = cis2_rsqrt(mean + eps);
    for (size_t i = 0; i < n; i++) {
        float scaled = x[i] * inv;
        out[i] = scaled * weight[i];
    }
}

/* ---- §6.6 table digest ---- */
static void feed_f32_le(cis2_sha256_ctx *ctx, uint32_t bits)
{
    uint8_t b[4];
    b[0] = (uint8_t)(bits & 0xFFu);
    b[1] = (uint8_t)((bits >> 8) & 0xFFu);
    b[2] = (uint8_t)((bits >> 16) & 0xFFu);
    b[3] = (uint8_t)((bits >> 24) & 0xFFu);
    cis2_sha256_update(ctx, b, 4);
}

void cis2_feed_table_digest(cis2_sha256_ctx *ctx)
{
    feed_f32_le(ctx, EXP_LOG2E_BITS);
    feed_f32_le(ctx, EXP_C1_BITS);
    feed_f32_le(ctx, EXP_C2_BITS);
    for (int i = 0; i < 6; i++) feed_f32_le(ctx, EXP_P_BITS[i]);
    feed_f32_le(ctx, TWO_PI_HI_BITS);
    feed_f32_le(ctx, TWO_PI_LO_BITS);
    for (int i = 0; i < 11; i++) feed_f32_le(ctx, SIN_COEF_BITS[i]);
    for (int i = 0; i < 11; i++) feed_f32_le(ctx, COS_COEF_BITS[i]);
    feed_f32_le(ctx, LOG_SQRTHF_BITS);
    feed_f32_le(ctx, LOG_Q1_BITS);
    feed_f32_le(ctx, LOG_Q2_BITS);
    for (int i = 0; i < 9; i++) feed_f32_le(ctx, LOG_P_BITS[i]);
}
