/* quant_ternary.c — see quant_ternary.h. */
#include <stdlib.h>
#include <math.h>
#include "quant_ternary.h"

static inline void pack_trit(uint8_t *packed, size_t t, int trit)
{
    uint8_t code = (trit == 1) ? 0x1 : (trit == -1) ? 0x2 : 0x0;
    size_t byte_idx = t >> 2;
    unsigned shift = (unsigned)((t & 3) * 2);
    packed[byte_idx] = (uint8_t)(packed[byte_idx] & ~(0x3u << shift));
    packed[byte_idx] = (uint8_t)(packed[byte_idx] | (code << shift));
}

static inline int unpack_trit(const uint8_t *packed, size_t t)
{
    size_t byte_idx = t >> 2;
    unsigned shift = (unsigned)((t & 3) * 2);
    uint8_t code = (uint8_t)((packed[byte_idx] >> shift) & 0x3u);
    if (code == 0x1) return 1;
    if (code == 0x2) return -1;
    return 0;
}

void cis2_quantize_matrix_ternary(const float *w, size_t out_features, size_t in_features, cis2_tqmat *out)
{
    out->out_features = out_features;
    out->in_features = in_features;
    size_t n_trits = out_features * in_features;
    size_t n_packed_bytes = (n_trits + 3) / 4;
    out->packed = calloc(n_packed_bytes ? n_packed_bytes : 1, sizeof(uint8_t));
    out->scale = malloc(out_features * sizeof(float));

    for (size_t o = 0; o < out_features; o++) {
        const float *row = w + o * in_features;
        double sum_abs = 0.0; /* left-to-right accumulation, fixed order */
        for (size_t i = 0; i < in_features; i++) sum_abs += fabs((double)row[i]);
        float mean_abs = in_features ? (float)(sum_abs / (double)in_features) : 0.0f;
        float scale = (mean_abs > 0.0f) ? mean_abs : 1.0f;
        out->scale[o] = scale;
        float threshold = 0.5f * mean_abs;
        size_t base = o * in_features;
        for (size_t i = 0; i < in_features; i++) {
            float v = row[i];
            int trit = 0;
            if (v > threshold) trit = 1;
            else if (v < -threshold) trit = -1;
            pack_trit(out->packed, base + i, trit);
        }
    }
}

void cis2_tqmat_free(cis2_tqmat *m)
{
    if (!m) return;
    free(m->packed);
    free(m->scale);
    m->packed = NULL;
    m->scale = NULL;
}

void cis2_matvec_ternary(const cis2_tqmat *qw, const int8_t *qx, float x_scale, float *y)
{
    size_t out_features = qw->out_features;
    size_t in_features = qw->in_features;
    for (size_t o = 0; o < out_features; o++) {
        size_t base = o * in_features;
        int32_t acc = 0; /* pure int32 accumulation, fixed left-to-right order */
        for (size_t i = 0; i < in_features; i++) {
            int trit = unpack_trit(qw->packed, base + i);
            acc += (int32_t)trit * (int32_t)qx[i];
        }
        y[o] = (float)acc * qw->scale[o] * x_scale;
    }
}
