/* model.c — CIS-2 v0.2 model load + forward pass + witness/argmax digests. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "mathpin.h"
#include "json.h"
#include "sha256.h"

int cis2_sha256_file(const char *path, uint8_t out[32])
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    cis2_sha256_ctx ctx;
    cis2_sha256_init(&ctx);
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        cis2_sha256_update(&ctx, buf, n);
    }
    fclose(f);
    cis2_sha256_final(&ctx, out);
    return 0;
}

static char *read_whole_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    if (out_len) *out_len = rd;
    return buf;
}

/* ---- safetensors ---- */

typedef struct {
    char *raw; size_t raw_len;
    const uint8_t *data; size_t data_len;
    json_value *header;
} st_file;

static st_file g_st;

static void st_open(const char *path)
{
    size_t len;
    g_st.raw = read_whole_file(path, &len);
    if (!g_st.raw) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: cannot open %s\n", path);
        exit(95);
    }
    g_st.raw_len = len;
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= ((uint64_t)(uint8_t)g_st.raw[i]) << (8 * i);
    char *header_json = malloc(header_len + 1);
    memcpy(header_json, g_st.raw + 8, header_len);
    header_json[header_len] = '\0';
    g_st.header = json_parse(header_json);
    free(header_json);
    g_st.data = (const uint8_t *)(g_st.raw + 8 + header_len);
    g_st.data_len = g_st.raw_len - 8 - header_len;
}

static float *st_get_tensor(const char *name, size_t expect_numel)
{
    json_value *ent = json_obj_get(g_st.header, name);
    if (!ent) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: tensor '%s' not found in safetensors header\n", name);
        exit(95);
    }
    json_value *dtype = json_obj_get(ent, "dtype");
    const char *dt = json_as_cstring(dtype);
    if (!dt || strcmp(dt, "BF16") != 0) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: tensor '%s' dtype is not BF16\n", name);
        exit(95);
    }
    json_value *offsets = json_obj_get(ent, "data_offsets");
    long start = (long)json_as_number(json_array_get(offsets, 0));
    long end = (long)json_as_number(json_array_get(offsets, 1));
    size_t nbytes = (size_t)(end - start);
    size_t numel = nbytes / 2;
    if (expect_numel != 0 && numel != expect_numel) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: tensor '%s' numel mismatch (got %zu expected %zu)\n",
                name, numel, expect_numel);
        exit(95);
    }
    const uint8_t *src = g_st.data + start;
    float *out = malloc(numel * sizeof(float));
    for (size_t i = 0; i < numel; i++) {
        uint16_t b = (uint16_t)src[2 * i] | ((uint16_t)src[2 * i + 1] << 8);
        uint32_t bits = ((uint32_t)b) << 16;
        out[i] = cis2_f32_from_bits(bits);
    }
    return out;
}

static float *st_get_tensor_opt(const char *name, size_t expect_numel)
{
    json_value *ent = json_obj_get(g_st.header, name);
    if (!ent) return NULL;
    return st_get_tensor(name, expect_numel);
}

/* ---- config ---- */

static float json_double_to_f32(json_value *v, float fallback)
{
    if (!v) return fallback;
    double d = json_as_number(v);
    return (float)d; /* single RNE narrowing, matches spec §2.3 */
}

cis2_model *cis2_model_load(const char *safetensors_path, const char *config_json_text)
{
    json_value *cfgroot = json_parse(config_json_text);
    if (!cfgroot) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: failed to parse config.json\n");
        exit(95);
    }

    cis2_model *m = calloc(1, sizeof(cis2_model));
    m->cfg.hidden_size = (size_t)json_as_number(json_obj_get(cfgroot, "hidden_size"));
    m->cfg.intermediate_size = (size_t)json_as_number(json_obj_get(cfgroot, "intermediate_size"));
    m->cfg.num_hidden_layers = (size_t)json_as_number(json_obj_get(cfgroot, "num_hidden_layers"));
    m->cfg.num_attention_heads = (size_t)json_as_number(json_obj_get(cfgroot, "num_attention_heads"));
    m->cfg.num_key_value_heads = (size_t)json_as_number(json_obj_get(cfgroot, "num_key_value_heads"));
    m->cfg.rms_norm_eps = json_double_to_f32(json_obj_get(cfgroot, "rms_norm_eps"), cis2_f32_from_bits(CIS2_EPS_F32_BITS));
    json_value *theta_v = json_obj_get(cfgroot, "rope_theta");
    m->cfg.rope_theta = theta_v ? (float)json_as_number(theta_v) : 100000.0f;
    json_value *tie_v = json_obj_get(cfgroot, "tie_word_embeddings");
    m->cfg.tie_word_embeddings = (tie_v && tie_v->type == JSON_BOOL) ? tie_v->u.boolean : 1;
    m->cfg.vocab_size = (size_t)json_as_number(json_obj_get(cfgroot, "vocab_size"));
    json_free(cfgroot);

    m->head_dim = m->cfg.hidden_size / m->cfg.num_attention_heads;
    m->half = m->head_dim / 2;
    m->group = m->cfg.num_attention_heads / m->cfg.num_key_value_heads;

    st_open(safetensors_path);

    size_t hidden = m->cfg.hidden_size;
    size_t inter = m->cfg.intermediate_size;
    size_t nkv_dim = m->cfg.num_key_value_heads * m->head_dim;
    size_t vocab = m->cfg.vocab_size;

    m->embed_tokens = st_get_tensor("model.embed_tokens.weight", vocab * hidden);
    m->norm_weight = st_get_tensor("model.norm.weight", hidden);

    if (!m->cfg.tie_word_embeddings) {
        m->lm_head = st_get_tensor_opt("lm_head.weight", vocab * hidden);
    }

    m->layers = calloc(m->cfg.num_hidden_layers, sizeof(cis2_layer));
    char name[256];
    for (size_t i = 0; i < m->cfg.num_hidden_layers; i++) {
        cis2_layer *L = &m->layers[i];
        snprintf(name, sizeof(name), "model.layers.%zu.input_layernorm.weight", i);
        L->input_layernorm = st_get_tensor(name, hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.post_attention_layernorm.weight", i);
        L->post_attention_layernorm = st_get_tensor(name, hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.q_proj.weight", i);
        L->q_proj = st_get_tensor(name, hidden * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.k_proj.weight", i);
        L->k_proj = st_get_tensor(name, nkv_dim * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.v_proj.weight", i);
        L->v_proj = st_get_tensor(name, nkv_dim * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.o_proj.weight", i);
        L->o_proj = st_get_tensor(name, hidden * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.mlp.gate_proj.weight", i);
        L->gate_proj = st_get_tensor(name, inter * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.mlp.up_proj.weight", i);
        L->up_proj = st_get_tensor(name, inter * hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.mlp.down_proj.weight", i);
        L->down_proj = st_get_tensor(name, hidden * inter);

        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.q_proj.bias", i);
        L->q_bias = st_get_tensor_opt(name, hidden);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.k_proj.bias", i);
        L->k_bias = st_get_tensor_opt(name, nkv_dim);
        snprintf(name, sizeof(name), "model.layers.%zu.self_attn.v_proj.bias", i);
        L->v_bias = st_get_tensor_opt(name, nkv_dim);
    }

    /* §7.1 inv_freq */
    m->inv_freq = malloc(m->half * sizeof(float));
    float ln_theta = cis2_ln_pinned(m->cfg.rope_theta);
    for (size_t i = 0; i < m->half; i++) {
        float frac = ((float)(2 * i)) / (float)m->head_dim;
        float neg_arg = -(frac * ln_theta);
        m->inv_freq[i] = cis2_exp_pinned(neg_arg);
    }

    free(g_st.raw);
    json_free(g_st.header);
    memset(&g_st, 0, sizeof(g_st));

    return m;
}

void cis2_model_free(cis2_model *m)
{
    if (!m) return;
    free(m->embed_tokens);
    free(m->norm_weight);
    free(m->lm_head);
    free(m->inv_freq);
    for (size_t i = 0; i < m->cfg.num_hidden_layers; i++) {
        cis2_layer *L = &m->layers[i];
        free(L->input_layernorm); free(L->post_attention_layernorm);
        free(L->q_proj); free(L->k_proj); free(L->v_proj); free(L->o_proj);
        free(L->gate_proj); free(L->up_proj); free(L->down_proj);
        free(L->q_bias); free(L->k_bias); free(L->v_bias);
    }
    free(m->layers);
    free(m);
}

/* ---- forward pass ---- */

static void add_bias_opt(float *x, const float *bias, size_t n)
{
    if (!bias) return;
    for (size_t i = 0; i < n; i++) x[i] = x[i] + bias[i]; /* §5.4 */
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

/* E15k debug-only: SHA-256 over the raw LE f32 bit patterns of a slice,
 * printed as a `CIS2_DUMP` line to stderr in the same format the Rust
 * reference's matching side-channel produces, so the two can be diffed
 * line-for-line. Not part of any normative digest; gated on `g_dump_layers`
 * (env var CIS2_DUMP_LAYERS) and only ever called for the single position
 * that produces step 0's logits. */
static int g_dump_layers = 0;
static int g_dump_run_count = 0; /* only dump on the first cis2_run_decode() call (r1), to match Rust's run1-only gating */

void cis2_set_dump_layers(int enabled) { g_dump_layers = enabled; }

static void dump_line(const char *layer, const char *field, size_t pos, const float *xs, size_t n)
{
    cis2_sha256_ctx ctx;
    cis2_sha256_init(&ctx);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = cis2_f32_bits(xs[i]);
        uint8_t b[4] = { (uint8_t)(bits & 0xFF), (uint8_t)((bits >> 8) & 0xFF),
                         (uint8_t)((bits >> 16) & 0xFF), (uint8_t)((bits >> 24) & 0xFF) };
        cis2_sha256_update(&ctx, b, 4);
    }
    uint8_t digest[32];
    cis2_sha256_final(&ctx, digest);
    char hex[65];
    static const char *hexd = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        hex[2 * i] = hexd[(digest[i] >> 4) & 0xF];
        hex[2 * i + 1] = hexd[digest[i] & 0xF];
    }
    hex[64] = '\0';
    fprintf(stderr, "CIS2_DUMP layer=%s field=%s pos=%zu digest=%s\n", layer, field, pos, hex);
}

/* Runs a full forward pass over tokens[0..ntok-1], returns logits[vocab]
 * for the LAST position only (malloc'd, caller frees). If `dump_last` is
 * nonzero and g_dump_layers is enabled, emits per-layer CIS2_DUMP lines
 * (E15k) for position ntok-1 only (the forward pass that produces step
 * 0's logits). */
static float *forward_full(const cis2_model *m, const uint32_t *tokens, size_t ntok, int dump_last)
{
    int dump = g_dump_layers && dump_last;
    size_t dump_pos = ntok - 1;
    size_t hidden = m->cfg.hidden_size;
    size_t inter = m->cfg.intermediate_size;
    size_t head_dim = m->head_dim;
    size_t half = m->half;
    size_t n_heads = m->cfg.num_attention_heads;
    size_t n_kv = m->cfg.num_key_value_heads;
    size_t nkv_dim = n_kv * head_dim;
    size_t group = m->group;
    float eps = m->cfg.rms_norm_eps;

    float *h = malloc(ntok * hidden * sizeof(float));
    for (size_t p = 0; p < ntok; p++) {
        memcpy(h + p * hidden, m->embed_tokens + (size_t)tokens[p] * hidden, hidden * sizeof(float));
    }
    if (dump) dump_line("embed", "token_embedding", dump_pos, h + dump_pos * hidden, hidden);

    /* per-position RoPE cos/sin tables, shared across all layers/heads (§7.3) */
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
    float scale = cis2_rsqrt((float)head_dim);

    for (size_t layer_i = 0; layer_i < m->cfg.num_hidden_layers; layer_i++) {
        const cis2_layer *L = &m->layers[layer_i];

        for (size_t p = 0; p < ntok; p++)
            cis2_rmsnorm(h + p * hidden, L->input_layernorm, eps, ln1 + p * hidden, hidden);

        for (size_t p = 0; p < ntok; p++) {
            cis2_matvec(L->q_proj, ln1 + p * hidden, q + p * hidden, hidden, hidden);
            add_bias_opt(q + p * hidden, L->q_bias, hidden);
            cis2_matvec(L->k_proj, ln1 + p * hidden, k + p * nkv_dim, nkv_dim, hidden);
            add_bias_opt(k + p * nkv_dim, L->k_bias, nkv_dim);
            cis2_matvec(L->v_proj, ln1 + p * hidden, v + p * nkv_dim, nkv_dim, hidden);
            add_bias_opt(v + p * nkv_dim, L->v_bias, nkv_dim);

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
                    scores[j] = d * scale;
                }
                /* softmax over scores[0..p] */
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
            cis2_matvec(L->o_proj, attn_out + p * hidden, o + p * hidden, hidden, hidden);
            for (size_t i = 0; i < hidden; i++) h[p * hidden + i] = h[p * hidden + i] + o[p * hidden + i];
        }
        if (dump) {
            char label[32];
            snprintf(label, sizeof(label), "block%zu", layer_i);
            dump_line(label, "post_attention_hidden", dump_pos, h + dump_pos * hidden, hidden);
        }

        for (size_t p = 0; p < ntok; p++)
            cis2_rmsnorm(h + p * hidden, L->post_attention_layernorm, eps, ln2 + p * hidden, hidden);

        for (size_t p = 0; p < ntok; p++) {
            cis2_matvec(L->gate_proj, ln2 + p * hidden, gate + p * inter, inter, hidden);
            cis2_matvec(L->up_proj, ln2 + p * hidden, up + p * inter, inter, hidden);
            for (size_t i = 0; i < inter; i++) {
                hid[p * inter + i] = cis2_silu_pinned(gate[p * inter + i]) * up[p * inter + i];
            }
            cis2_matvec(L->down_proj, hid + p * inter, down + p * hidden, hidden, inter);
            for (size_t i = 0; i < hidden; i++) h[p * hidden + i] = h[p * hidden + i] + down[p * hidden + i];
        }
        if (dump) {
            char label[32];
            snprintf(label, sizeof(label), "block%zu", layer_i);
            dump_line(label, "post_mlp_hidden", dump_pos, h + dump_pos * hidden, hidden);
        }
    }

    size_t last = ntok - 1;
    float *hn = malloc(hidden * sizeof(float));
    cis2_rmsnorm(h + last * hidden, m->norm_weight, eps, hn, hidden);
    if (dump) dump_line("final_norm", "final_norm_output", dump_pos, hn, hidden);
    const float *lm_w = m->cfg.tie_word_embeddings ? m->embed_tokens : (m->lm_head ? m->lm_head : m->embed_tokens);
    float *logits = malloc(m->cfg.vocab_size * sizeof(float));
    cis2_matvec(lm_w, hn, logits, m->cfg.vocab_size, hidden);
    if (dump) dump_line("logits", "pre_argmax_logits", dump_pos, logits, m->cfg.vocab_size);

    free(hn);
    free(h); free(cos_tab); free(sin_tab);
    free(ln1); free(ln2); free(q); free(k); free(v);
    free(attn_out); free(o); free(gate); free(up); free(hid); free(down); free(scores);

    return logits;
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

void cis2_run_decode(const cis2_model *m,
                       const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen,
                       const uint8_t weights_sha256[32],
                       const uint8_t tokenizer_sha256[32],
                       const uint8_t config_sha256[32],
                       cis2_run_result *out)
{
    g_dump_run_count++;

    /* §6.6 table digest */
    cis2_sha256_ctx table_ctx;
    cis2_sha256_init(&table_ctx);
    cis2_feed_table_digest(&table_ctx);
    cis2_sha256_final(&table_ctx, out->table_digest);

    /* §7.2 inv_freq table digest */
    cis2_sha256_ctx invfreq_ctx;
    cis2_sha256_init(&invfreq_ctx);
    for (size_t i = 0; i < m->half; i++) {
        uint32_t bits = cis2_f32_bits(m->inv_freq[i]);
        uint8_t b[4] = { (uint8_t)(bits & 0xFF), (uint8_t)((bits >> 8) & 0xFF),
                         (uint8_t)((bits >> 16) & 0xFF), (uint8_t)((bits >> 24) & 0xFF) };
        cis2_sha256_update(&invfreq_ctx, b, 4);
    }
    cis2_sha256_final(&invfreq_ctx, out->inv_freq_table_digest);

    /* §12.1 witness chain */
    cis2_sha256_ctx witness;
    cis2_sha256_init(&witness);
    /* E15k: item order corrected to match the *actual* Rust reference
     * behavior (src/main.rs), which is the source of truth for the pinned
     * CIS2_REF=a0c563ef... test vector (reconfirmed by the 20-cell
     * E15d(a) CI matrix). docs/CIS2_SPEC_v0.2.md §12.1's prose lists this
     * as table_digest, inv_freq_table_digest, weights, tokenizer, config
     * — but that prose does not match the reference binary that produced
     * its own pinned vector. This was the entire witness-digest mismatch:
     * every input byte-run was already bit-identical (E15k dump diff,
     * steps 0-1); only the SHA-256 update *order* of these five 32-byte
     * inputs differed, which is enough to produce a completely unrelated
     * final digest despite identical logits/tokens throughout. */
    cis2_sha256_update(&witness, weights_sha256, 32);
    cis2_sha256_update(&witness, tokenizer_sha256, 32);
    cis2_sha256_update(&witness, config_sha256, 32);
    cis2_sha256_update(&witness, out->table_digest, 32);
    cis2_sha256_update(&witness, out->inv_freq_table_digest, 32);
    for (size_t i = 0; i < n_prompt; i++) feed_u32_le(&witness, prompt_ids[i]);

    /* §12.2 argmax-token digest (separate instance) */
    cis2_sha256_ctx argctx;
    cis2_sha256_init(&argctx);
    for (size_t i = 0; i < n_prompt; i++) feed_u32_le(&argctx, prompt_ids[i]);

    size_t cap = n_prompt + n_gen;
    uint32_t *tokens = malloc(cap * sizeof(uint32_t));
    memcpy(tokens, prompt_ids, n_prompt * sizeof(uint32_t));
    size_t ntok = n_prompt;

    uint32_t *generated = malloc(n_gen * sizeof(uint32_t));

    for (size_t step = 0; step < n_gen; step++) {
        float *logits = forward_full(m, tokens, ntok, g_dump_run_count == 1 && step <= 1);
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
