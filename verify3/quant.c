/* quant.c — see quant.h. */
#include <stdlib.h>
#include <math.h>
#include "quant.h"

static int8_t clamp_i8(long v)
{
    if (v > 127) return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

void cis2_quantize_matrix(const float *w, size_t out_features, size_t in_features, cis2_qmat *out)
{
    out->out_features = out_features;
    out->in_features = in_features;
    out->q = malloc(out_features * in_features * sizeof(int8_t));
    out->scale = malloc(out_features * sizeof(float));
    for (size_t o = 0; o < out_features; o++) {
        const float *row = w + o * in_features;
        float absmax = 0.0f;
        for (size_t i = 0; i < in_features; i++) {
            float a = fabsf(row[i]);
            if (a > absmax) absmax = a;
        }
        float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        out->scale[o] = scale;
        int8_t *qrow = out->q + o * in_features;
        for (size_t i = 0; i < in_features; i++) {
            float scaled = row[i] / scale;
            long rounded = lroundf(scaled);
            qrow[i] = clamp_i8(rounded);
        }
    }
}

void cis2_qmat_free(cis2_qmat *m)
{
    if (!m) return;
    free(m->q);
    free(m->scale);
    m->q = NULL;
    m->scale = NULL;
}

void cis2_quantize_vec(const float *x, size_t n, int8_t *qx, float *scale_out)
{
    float absmax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > absmax) absmax = a;
    }
    float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    *scale_out = scale;
    for (size_t i = 0; i < n; i++) {
        long rounded = lroundf(x[i] / scale);
        qx[i] = clamp_i8(rounded);
    }
}

void cis2_matvec_int8(const cis2_qmat *qw, const int8_t *qx, float x_scale, float *y)
{
    size_t out_features = qw->out_features;
    size_t in_features = qw->in_features;
    for (size_t o = 0; o < out_features; o++) {
        const int8_t *row = qw->q + o * in_features;
        int32_t acc = 0; /* pure int32 accumulation, fixed left-to-right order */
        for (size_t i = 0; i < in_features; i++) {
            acc += (int32_t)row[i] * (int32_t)qx[i];
        }
        y[o] = (float)acc * qw->scale[o] * x_scale;
    }
}
