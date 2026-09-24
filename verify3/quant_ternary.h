/* quant_ternary.h — CIS-2 ternary (BitNet-b1.58-style) per-output-channel
 * weight quantization + integer-accumulation matvec, added for QUEUE item
 * small-2 (TERNARY vs INT8 AT MATCHED BYTES). Purely additive alongside
 * quant.h/quant.c (int8 path, small-1) — does not modify them.
 *
 * Quantization scheme (prior art: BitNet / BitNet b1.58, bitnet.cpp I2_S
 * packing convention; not claimed as novel here):
 *   scale[o]    = mean_i(|w[o,i]|)                 (absmean, per row/channel)
 *   threshold   = 0.5 * scale[o]
 *   trit[o,i]   = +1 if w[o,i] >  threshold
 *               = -1 if w[o,i] < -threshold
 *               =  0 otherwise
 * Weights are packed 2 bits/trit, 4 trits/byte (a simple from-scratch
 * packer, not the alice-aegis packed format — kept simple per task scope).
 * Trit codes: 0b00 = 0, 0b01 = +1, 0b10 = -1 (0b11 unused).
 *
 * Discipline: same as quant.c's int8 path — the matvec reduction is pure
 * int32 arithmetic (trit values are -1/0/+1, multiplied against int8
 * activations), fixed left-to-right accumulation order, bit-exact by
 * construction; only the final per-output dequant is a single deterministic
 * float multiply (acc * scale[o] * x_scale), not accumulated.
 */
#ifndef CIS2_V3_QUANT_TERNARY_H
#define CIS2_V3_QUANT_TERNARY_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t *packed; /* ceil(out_features*in_features/4) bytes; 2 bits/trit,
                         flat row-major trit index t = o*in_features + i,
                         byte = packed[t/4], code = (byte >> ((t%4)*2)) & 3 */
    float *scale;    /* [out_features], absmean dequant scale per row */
    size_t out_features;
    size_t in_features;
} cis2_tqmat;

void cis2_quantize_matrix_ternary(const float *w, size_t out_features, size_t in_features, cis2_tqmat *out);
void cis2_tqmat_free(cis2_tqmat *m);

/* y[o] = (int32 dot(trit(qw[o,:]) in {-1,0,1}, qx), fixed left-to-right
 *        accumulation) * qw->scale[o] * x_scale.
 * Reuses cis2_quantize_vec (quant.h, small-1) for int8 activation
 * quantization — unmodified, shared, not a new activation scheme. */
void cis2_matvec_ternary(const cis2_tqmat *qw, const int8_t *qx, float x_scale, float *y);

#endif
