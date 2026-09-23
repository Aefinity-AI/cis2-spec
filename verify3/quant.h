/* quant.h — CIS-2 int8 per-channel weight quantization + integer-accumulation
 * matvec, added for QUEUE item small-1 (INTEGER-ONLY DETERMINISTIC PATH).
 *
 * Discipline: the reduction itself (the matmul dot product) is pure int32
 * arithmetic, fixed left-to-right accumulation order, so it is bit-exact by
 * construction on any conforming integer ALU. Quantization (float -> int8)
 * happens once at model-load time, not in the per-token hot path; the
 * final per-output-channel dequantization is a single deterministic float
 * multiply (no reassociation, same discipline as the fp32 reference).
 */
#ifndef CIS2_V3_QUANT_H
#define CIS2_V3_QUANT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int8_t *q;     /* [out_features, in_features], row-major */
    float *scale;  /* [out_features], dequant scale per output channel */
    size_t out_features;
    size_t in_features;
} cis2_qmat;

/* Per-output-channel (row) int8 quantization:
 *   scale[o]  = max_i(|w[o,i]|) / 127.0f   (0 if row is all-zero -> scale 1.0)
 *   qw[o,i]   = clamp(round(w[o,i] / scale[o]), -127, 127)
 * Single left-to-right pass per row. Deterministic given deterministic
 * float rounding (which IEEE-754 roundf() is, on a given machine).
 */
void cis2_quantize_matrix(const float *w, size_t out_features, size_t in_features, cis2_qmat *out);
void cis2_qmat_free(cis2_qmat *m);

/* Per-vector int8 activation quantization: single scale over the whole
 * vector (absmax/127), deterministic fixed left-to-right order. */
void cis2_quantize_vec(const float *x, size_t n, int8_t *qx, float *scale_out);

/* y[o] = (int32 dot(qw[o,:], qx), fixed left-to-right accumulation)
 *        * qw->scale[o] * x_scale
 * The reduction is pure int32 arithmetic (bit-exact by construction);
 * only the final per-output rescale is a single deterministic float
 * multiply, done once per output element (not accumulated).
 */
void cis2_matvec_int8(const cis2_qmat *qw, const int8_t *qx, float x_scale, float *y);

#endif
