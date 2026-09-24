/* model_ternary.c — see model_ternary.h. Mirrors model_int8.c's structure
 * (small-1) with cis2_tqmat/cis2_matvec_ternary in place of
 * cis2_qmat/cis2_matvec_int8. Activation quantization reuses quant.c's
 * cis2_quantize_vec (small-1, unmodified) — only the weight side is new.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model_ternary.h"
#include "quant.h"
#include "mathpin.h"
#include "sha256.h"

cis2_tqmodel *cis2_tqmodel_build(const cis2_model *m)
{
    cis2_tqmodel *qm = calloc(1, sizeof(cis2_tqmodel));
    qm->base = m;
    size_t hidden = m->cfg.hidden_size;
    size_t inter = m->cfg.intermediate_size;
    size_t nkv_dim = m->cfg.num_key_value_heads * m->head_dim;
    size_t vocab = m->cfg.vocab_size;

    cis2_quantize_matrix_ternary(m->embed_tokens, vocab, hidden, &qm->embed);
    qm->has_separate_lm_head = !m->cfg.tie_word_embeddings && m->lm_head != NULL;
    if (qm->has_separate_lm_head) {
        cis2_quantize_matrix_ternary(m->lm_head, vocab, hidden, &qm->lm_head);
    }

    qm->layers = calloc(m->cfg.num_hidden_layers, sizeof(cis2_tqlayer));
    for (size_t i = 0; i < m->cfg.num_hidden_layers; i++) {
        const cis2_layer *L = &m->layers[i];
        cis2_tqlayer *QL = &qm->layers[i];
        cis2_quantize_matrix_ternary(L->q_proj, hidden, hidden, &QL->q_proj);
        cis2_quantize_matrix_ternary(L->k_proj, nkv_dim, hidden, &QL->k_proj);
        cis2_quantize_matrix_ternary(L->v_proj, nkv_dim, hidden, &QL->v_proj);
        cis2_quantize_matrix_ternary(L->o_proj, hidden, hidden, &QL->o_proj);
        cis2_quantize_matrix_ternary(L->gate_proj, inter, hidden, &QL->gate_proj);
        cis2_quantize_matrix_ternary(L->up_proj, inter, hidden, &QL->up_proj);
        cis2_quantize_matrix_ternary(L->down_proj, hidden, inter, &QL->down_proj);
    }

    /* size accounting: packed trits (2 bits/trit, ceil(out*in/4) bytes) +
     * per-row scale floats, matching small-1's nb accounting shape. */
    size_t nb = (qm->embed.out_features * qm->embed.in_features + 3) / 4
              + qm->embed.out_features * sizeof(float);
    if (qm->has_separate_lm_head)
        nb += (qm->lm_head.out_features * qm->lm_head.in_features + 3) / 4
            + qm->lm_head.out_features * sizeof(float);
    for (size_t i = 0; i < m->cfg.num_hidden_layers; i++) {
        cis2_tqlayer *QL = &qm->layers[i];
        cis2_tqmat *mats[7] = { &QL->q_proj, &QL->k_proj, &QL->v_proj, &QL->o_proj,
                                 &QL->gate_proj, &QL->up_proj, &QL->down_proj };
        for (int j = 0; j < 7; j++) {
            nb += (mats[j]->out_features * mats[j]->in_features + 3) / 4
                + mats[j]->out_features * sizeof(float);
        }
    }
    qm->n_bytes_quantized = nb;
    return qm;
}

void cis2_tqmodel_free(cis2_tqmodel *qm)
{
    if (!qm) return;
    cis2_tqmat_free(&qm->embed);
    if (qm->has_separate_lm_head) cis2_tqmat_free(&qm->lm_head);
    for (size_t i = 0; i < qm->base->cfg.num_hidden_layers; i++) {
        cis2_tqlayer *QL = &qm->layers[i];
        cis2_tqmat_free(&QL->q_proj); cis2_tqmat_free(&QL->k_proj);
        cis2_tqmat_free(&QL->v_proj); cis2_tqmat_free(&QL->o_proj);
        cis2_tqmat_free(&QL->gate_proj); cis2_tqmat_free(&QL->up_proj);
        cis2_tqmat_free(&QL->down_proj);
    }
    free(qm->layers);
    free(qm);
}

static void rope_apply_head(float *head, const float *cos_v, const float *sin_v, size_t half)
{
    float *out = malloc(2 * half * sizeof(float));
    for (size_t i = 0; i < half; i++) {
        float x1 = head[i];
        float x2 = head[i + half];
        float t1 = x1 * cos_v[i];
        float t2 = (-x2) * sin_v[i];
        out[i] = t1 + t2;
        float t3 = x2 * cos_v[i];
        float t4 = x1 * sin_v[i];
        out[i + half] = t3 + t4;
    }
    for (size_t i = 0; i < 2 * half; i++) head[i] = out[i];
    free(out);
}

/* qmatvec_dispatch: use_ternary -> cis2_matvec_ternary (int32 accumulation,
 * trits in {-1,0,1}); !use_ternary -> cis2_matvec (fp32, the untouched
 * reference path), mirrors model_int8.c's qmatvec dispatch shape. */
static void qmatvec(int use_ternary, const cis2_tqmat *qw, const float *w_fp32,
                     const float *x, float *y, size_t out_features, size_t in_features,
                     int8_t *qx_scratch)
{
    if (use_ternary) {
        float x_scale;
        cis2_quantize_vec(x, in_features, qx_scratch, &x_scale);
        cis2_matvec_ternary(qw, qx_scratch, x_scale, y);
    } else {
        cis2_matvec(w_fp32, x, y, out_features, in_features);
    }
}

typedef void (*logits_cb)(void *ctx, size_t pos, const float *logits, size_t vocab);

static void forward_generic(const cis2_model *m, const cis2_tqmodel *qm, int use_ternary,
                             const uint32_t *tokens, size_t ntok,
                             float **out_last_logits /* NULL if not wanted */,
                             logits_cb cb, void *cb_ctx)
{
    size_t hidden = m->cfg.hidden_size;
    size_t inter = m->cfg.intermediate_size;
    size_t head_dim = m->head_dim;
    size_t half = m->half;
    size_t n_heads = m->cfg.num_attention_heads;
    size_t n_kv = m->cfg.num_key_value_heads;
    size_t nkv_dim = n_kv * head_dim;
    size_t group = m->group;
    float eps = m->cfg.rms_norm_eps;
    size_t vocab = m->cfg.vocab_size;

    float *h = malloc(ntok * hidden * sizeof(float));
    for (size_t p = 0; p < ntok; p++) {
        if (use_ternary) {
            size_t base = (size_t)tokens[p] * hidden;
            float scale = qm->embed.scale[tokens[p]];
            for (size_t i = 0; i < hidden; i++) {
                size_t t = base + i;
                size_t byte_idx = t >> 2;
                unsigned shift = (unsigned)((t & 3) * 2);
                uint8_t code = (uint8_t)((qm->embed.packed[byte_idx] >> shift) & 0x3u);
                float tv = (code == 0x1) ? 1.0f : (code == 0x2) ? -1.0f : 0.0f;
                h[p * hidden + i] = tv * scale;
            }
        } else {
            memcpy(h + p * hidden, m->embed_tokens + (size_t)tokens[p] * hidden, hidden * sizeof(float));
        }
    }

    float *cos_tab = malloc(ntok * half * sizeof(float));
    float *sin_tab = malloc(ntok * half * sizeof(float));
    for (size_t p = 0; p < ntok; p++) {
        for (size_t i = 0; i < half; i++) {
            float angle = (float)p * m->inv_freq[i];
            cos_tab[p * half + i] = cis2_cos_pinned(angle);
            sin_tab[p * half + i] = cis2_sin_pinned(angle);
        }
    }

    float *ln1 = malloc(ntok * hidden * sizeof(float));
    float *ln2 = malloc(ntok * hidden * sizeof(float));
    float *q = malloc(ntok * hidden * sizeof(float));
    float *k = malloc(ntok * nkv_dim * sizeof(float));
    float *v = malloc(ntok * nkv_dim * sizeof(float));
    float *attn_out = malloc(ntok * hidden * sizeof(float));
    float *o = malloc(ntok * hidden * sizeof(float));
    float *gate = malloc(ntok * inter * sizeof(float));
    float *up = malloc(ntok * inter * sizeof(float));
    float *hid = malloc(ntok * inter * sizeof(float));
    float *down = malloc(ntok * hidden * sizeof(float));
    float *scores = malloc(ntok * sizeof(float));
    float scale_attn = cis2_rsqrt((float)head_dim);
    size_t max_in = inter > hidden ? inter : hidden;
    int8_t *qx_scratch = malloc(max_in * sizeof(int8_t));

    for (size_t layer_i = 0; layer_i < m->cfg.num_hidden_layers; layer_i++) {
        const cis2_layer *L = &m->layers[layer_i];
        const cis2_tqlayer *QL = use_ternary ? &qm->layers[layer_i] : NULL;

        for (size_t p = 0; p < ntok; p++)
            cis2_rmsnorm(h + p * hidden, L->input_layernorm, eps, ln1 + p * hidden, hidden);

        for (size_t p = 0; p < ntok; p++) {
            qmatvec(use_ternary, use_ternary ? &QL->q_proj : NULL, L->q_proj, ln1 + p * hidden, q + p * hidden, hidden, hidden, qx_scratch);
            if (L->q_bias) for (size_t i = 0; i < hidden; i++) q[p * hidden + i] += L->q_bias[i];
            qmatvec(use_ternary, use_ternary ? &QL->k_proj : NULL, L->k_proj, ln1 + p * hidden, k + p * nkv_dim, nkv_dim, hidden, qx_scratch);
            if (L->k_bias) for (size_t i = 0; i < nkv_dim; i++) k[p * nkv_dim + i] += L->k_bias[i];
            qmatvec(use_ternary, use_ternary ? &QL->v_proj : NULL, L->v_proj, ln1 + p * hidden, v + p * nkv_dim, nkv_dim, hidden, qx_scratch);
            if (L->v_bias) for (size_t i = 0; i < nkv_dim; i++) v[p * nkv_dim + i] += L->v_bias[i];

            for (size_t qh = 0; qh < n_heads; qh++)
                rope_apply_head(q + p * hidden + qh * head_dim, cos_tab + p * half, sin_tab + p * half, half);
            for (size_t kh = 0; kh < n_kv; kh++)
                rope_apply_head(k + p * nkv_dim + kh * head_dim, cos_tab + p * half, sin_tab + p * half, half);
        }

        for (size_t p = 0; p < ntok; p++) {
            for (size_t qh = 0; qh < n_heads; qh++) {
                size_t kv_head = qh / group;
                const float *q_head = q + p * hidden + qh * head_dim;
                for (size_t j = 0; j <= p; j++) {
                    const float *k_j = k + j * nkv_dim + kv_head * head_dim;
                    float d = cis2_dot_seq(q_head, k_j, head_dim);
                    scores[j] = d * scale_attn;
                }
                float max_v = scores[0];
                for (size_t j = 1; j <= p; j++) if (scores[j] > max_v) max_v = scores[j];
                for (size_t j = 0; j <= p; j++) scores[j] = cis2_exp_pinned(scores[j] - max_v);
                float denom = cis2_sum_seq(scores, p + 1);
                for (size_t j = 0; j <= p; j++) scores[j] = scores[j] / denom;

                float *out_head = attn_out + p * hidden + qh * head_dim;
                for (size_t d = 0; d < head_dim; d++) {
                    float acc = 0.0f;
                    for (size_t j = 0; j <= p; j++) {
                        float pv = scores[j] * v[j * nkv_dim + kv_head * head_dim + d];
                        acc = acc + pv;
                    }
                    out_head[d] = acc;
                }
            }
        }

        for (size_t p = 0; p < ntok; p++) {
            qmatvec(use_ternary, use_ternary ? &QL->o_proj : NULL, L->o_proj, attn_out + p * hidden, o + p * hidden, hidden, hidden, qx_scratch);
            for (size_t i = 0; i < hidden; i++) h[p * hidden + i] = h[p * hidden + i] + o[p * hidden + i];
        }

        for (size_t p = 0; p < ntok; p++)
            cis2_rmsnorm(h + p * hidden, L->post_attention_layernorm, eps, ln2 + p * hidden, hidden);

        for (size_t p = 0; p < ntok; p++) {
            qmatvec(use_ternary, use_ternary ? &QL->gate_proj : NULL, L->gate_proj, ln2 + p * hidden, gate + p * inter, inter, hidden, qx_scratch);
            qmatvec(use_ternary, use_ternary ? &QL->up_proj : NULL, L->up_proj, ln2 + p * hidden, up + p * inter, inter, hidden, qx_scratch);
            for (size_t i = 0; i < inter; i++) {
                hid[p * inter + i] = cis2_silu_pinned(gate[p * inter + i]) * up[p * inter + i];
            }
            qmatvec(use_ternary, use_ternary ? &QL->down_proj : NULL, L->down_proj, hid + p * inter, down + p * hidden, hidden, inter, qx_scratch);
            for (size_t i = 0; i < hidden; i++) h[p * hidden + i] = h[p * hidden + i] + down[p * hidden + i];
        }
    }

    float *hn = malloc(hidden * sizeof(float));
    float *logits = malloc(vocab * sizeof(float));
    const cis2_tqmat *qlm = use_ternary ? (qm->has_separate_lm_head ? &qm->lm_head : &qm->embed) : NULL;
    const float *lm_w = m->cfg.tie_word_embeddings ? m->embed_tokens : (m->lm_head ? m->lm_head : m->embed_tokens);

    if (cb) {
        for (size_t p = 0; p < ntok; p++) {
            cis2_rmsnorm(h + p * hidden, m->norm_weight, eps, hn, hidden);
            qmatvec(use_ternary, qlm, lm_w, hn, logits, vocab, hidden, qx_scratch);
            cb(cb_ctx, p, logits, vocab);
        }
    }
    if (out_last_logits) {
        size_t last = ntok - 1;
        cis2_rmsnorm(h + last * hidden, m->norm_weight, eps, hn, hidden);
        qmatvec(use_ternary, qlm, lm_w, hn, logits, vocab, hidden, qx_scratch);
        float *ret = malloc(vocab * sizeof(float));
        memcpy(ret, logits, vocab * sizeof(float));
        *out_last_logits = ret;
    }

    free(hn); free(logits); free(qx_scratch);
    free(h); free(cos_tab); free(sin_tab);
    free(ln1); free(ln2); free(q); free(k); free(v);
    free(attn_out); free(o); free(gate); free(up); free(hid); free(down); free(scores);
}

static uint32_t argmax_logits(const float *logits, size_t n)
{
    size_t best_idx = 0;
    float best_val = logits[0];
    for (size_t idx = 1; idx < n; idx++) {
        if (logits[idx] > best_val) { best_val = logits[idx]; best_idx = idx; }
    }
    return (uint32_t)best_idx;
}

static void feed_u32_le(cis2_sha256_ctx *ctx, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
                     (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF) };
    cis2_sha256_update(ctx, b, 4);
}

void cis2_tqrun_decode(const cis2_tqmodel *qm,
                        const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen,
                        const uint8_t weights_sha256[32],
                        const uint8_t tokenizer_sha256[32],
                        const uint8_t config_sha256[32],
                        cis2_tqrun_result *out)
{
    const cis2_model *m = qm->base;

    uint8_t table_digest[32], inv_freq_digest[32];
    cis2_sha256_ctx table_ctx;
    cis2_sha256_init(&table_ctx);
    cis2_feed_table_digest(&table_ctx);
    cis2_sha256_final(&table_ctx, table_digest);

    cis2_sha256_ctx invfreq_ctx;
    cis2_sha256_init(&invfreq_ctx);
    for (size_t i = 0; i < m->half; i++) {
        uint32_t bits = cis2_f32_bits(m->inv_freq[i]);
        uint8_t b[4] = { (uint8_t)(bits & 0xFF), (uint8_t)((bits >> 8) & 0xFF),
                         (uint8_t)((bits >> 16) & 0xFF), (uint8_t)((bits >> 24) & 0xFF) };
        cis2_sha256_update(&invfreq_ctx, b, 4);
    }
    cis2_sha256_final(&invfreq_ctx, inv_freq_digest);

    cis2_sha256_ctx witness;
    cis2_sha256_init(&witness);
    cis2_sha256_update(&witness, weights_sha256, 32);
    cis2_sha256_update(&witness, tokenizer_sha256, 32);
    cis2_sha256_update(&witness, config_sha256, 32);
    cis2_sha256_update(&witness, table_digest, 32);
    cis2_sha256_update(&witness, inv_freq_digest, 32);
    for (size_t i = 0; i < n_prompt; i++) feed_u32_le(&witness, prompt_ids[i]);

    cis2_sha256_ctx argctx;
    cis2_sha256_init(&argctx);
    for (size_t i = 0; i < n_prompt; i++) feed_u32_le(&argctx, prompt_ids[i]);

    size_t cap = n_prompt + n_gen;
    uint32_t *tokens = malloc(cap * sizeof(uint32_t));
    memcpy(tokens, prompt_ids, n_prompt * sizeof(uint32_t));
    size_t ntok = n_prompt;

    uint32_t *generated = malloc(n_gen * sizeof(uint32_t));

    for (size_t step = 0; step < n_gen; step++) {
        float *logits = NULL;
        forward_generic(m, qm, 1, tokens, ntok, &logits, NULL, NULL);
        for (size_t i = 0; i < m->cfg.vocab_size; i++) {
            uint32_t bits = cis2_f32_bits(logits[i]);
            uint8_t b[4] = { (uint8_t)(bits & 0xFF), (uint8_t)((bits >> 8) & 0xFF),
                             (uint8_t)((bits >> 16) & 0xFF), (uint8_t)((bits >> 24) & 0xFF) };
            cis2_sha256_update(&witness, b, 4);
        }
        uint32_t next = argmax_logits(logits, m->cfg.vocab_size);
        free(logits);
        feed_u32_le(&witness, next);
        feed_u32_le(&argctx, next);
        generated[step] = next;
        tokens[ntok] = next;
        ntok++;
    }

    cis2_sha256_final(&witness, out->witness_digest);
    cis2_sha256_final(&argctx, out->argmax_digest);
    out->generated_ids = generated;
    out->n_gen = n_gen;
    free(tokens);
}

typedef struct {
    const uint32_t *tokens;
    size_t ntok;
    double nll_sum;
    size_t top1_matches;
    uint32_t *pred_ids_out;
} eval_ctx;

static void eval_cb(void *vctx, size_t pos, const float *logits, size_t vocab)
{
    eval_ctx *ctx = vctx;
    if (pos + 1 >= ctx->ntok) return;
    uint32_t target = ctx->tokens[pos + 1];
    uint32_t pred = argmax_logits(logits, vocab);
    if (pred == target) ctx->top1_matches++;
    if (ctx->pred_ids_out) ctx->pred_ids_out[pos] = pred;

    float max_v = logits[0];
    for (size_t i = 1; i < vocab; i++) if (logits[i] > max_v) max_v = logits[i];
    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab; i++) sum_exp += exp((double)(logits[i] - max_v));
    double log_sum_exp = log(sum_exp);
    double log_p_target = (double)(logits[target] - max_v) - log_sum_exp;
    ctx->nll_sum += -log_p_target;
}

void cis2_eval_teacher_forced_ternary(const cis2_model *m, const cis2_tqmodel *qm, int use_ternary,
                                       const uint32_t *tokens, size_t ntok,
                                       double *nll_sum_out, size_t *top1_matches_out,
                                       uint32_t *pred_ids_out)
{
    eval_ctx ctx = { tokens, ntok, 0.0, 0, pred_ids_out };
    forward_generic(m, qm, use_ternary, tokens, ntok, NULL, eval_cb, &ctx);
    *nll_sum_out = ctx.nll_sum;
    *top1_matches_out = ctx.top1_matches;
}
