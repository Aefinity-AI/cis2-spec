/* mathpin.h — CIS-2 v0.2 pinned fp32 primitives (spec §1, §5, §6). */
#ifndef CIS2_V3_MATHPIN_H
#define CIS2_V3_MATHPIN_H

#include <stdint.h>
#include <stddef.h>
#include "sha256.h"

/* §1.3 FTZ/DAZ pin + readback assert. Aborts (fprintf+exit) if not settable. */
void cis2_pin_ftz_daz(void);

/* §1.3 adversarial self-test. Aborts if it fails. */
void cis2_ftz_daz_selftest(void);

/* §6.1 */
float cis2_rsqrt(float x);

/* §6.2 */
float cis2_exp_pinned(float x);

/* §6.3 */
float cis2_sin_pinned(float x);
float cis2_cos_pinned(float x);

/* §6.4 */
float cis2_silu_pinned(float x);

/* §6.5 */
float cis2_ln_pinned(float x);

/* §5.1 */
float cis2_dot_seq(const float *a, const float *b, size_t n);

/* §5.3 */
float cis2_sum_seq(const float *a, size_t n);

/* §5.2: y[o] = dot_seq(w[o,:], x), w row-major [out_features, in_features] */
void cis2_matvec(const float *w, const float *x, float *y, size_t out_features, size_t in_features);

/* §8 RMSNorm. out may alias x. */
void cis2_rmsnorm(const float *x, const float *weight, float eps, float *out, size_t n);

/* §6.6: feed the pinned coefficient table (exp/sin/cos/ln consts) into ctx,
 * in the exact declared order, LE bytes of each f32 bit pattern. */
void cis2_feed_table_digest(cis2_sha256_ctx *ctx);

/* bit-pattern helpers */
static inline uint32_t cis2_f32_bits(float x) {
    uint32_t b;
    __builtin_memcpy(&b, &x, sizeof(b));
    return b;
}
static inline float cis2_f32_from_bits(uint32_t b) {
    float x;
    __builtin_memcpy(&x, &b, sizeof(x));
    return x;
}
/* spec v0.3 §6.3: f64-staged pi reduction needs an f64 bit-pattern helper. */
static inline double cis2_f64_from_bits(uint64_t b) {
    double x;
    __builtin_memcpy(&x, &b, sizeof(x));
    return x;
}

#define CIS2_EPS_F32_BITS 0x3727C5ACu

#endif
