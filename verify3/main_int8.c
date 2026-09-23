/* main_int8.c — driver for QUEUE item small-1 (INTEGER-ONLY DETERMINISTIC
 * PATH): int8 per-channel weight quantization + int32-accumulation matmul,
 * built on top of the existing fp32 cis2_model loader. Reports:
 *   - int8 witness digest, reproduced across two runs in-process AND
 *     (if invoked twice as a process) across two separate process runs
 *   - model size: fp32 tensor bytes vs int8-quantized bytes
 *   - tok/s for int8 greedy decode
 *   - perplexity (mean NLL, exp of it) and top-1 token agreement % between
 *     fp32 and int8 on a fixed teacher-forced text slice
 *
 * Usage: cis2_verify3_int8 <safetensors> <config.json> <tokenizer.json> \
 *          <eval_text_file> <eval_text_byte_offset> <eval_text_byte_len> \
 *          <max_eval_tokens>
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mathpin.h"
#include "model.h"
#include "model_int8.h"
#include "tokenizer.h"

static char *read_whole_file_or_die(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); exit(90); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

static void hex_encode(const uint8_t *bytes, size_t n, char *out)
{
    static const char *hexd = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = hexd[(bytes[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[bytes[i] & 0xF];
    }
    out[2 * n] = '\0';
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 8) {
        fprintf(stderr, "usage: %s weights.safetensors config.json tokenizer.json eval_text_file byte_offset byte_len max_eval_tokens\n", argv[0]);
        return 2;
    }
    const char *weights_path = argv[1];
    const char *config_path = argv[2];
    const char *tokenizer_path = argv[3];
    const char *eval_text_path = argv[4];
    long eval_offset = atol(argv[5]);
    long eval_len = atol(argv[6]);
    size_t max_eval_tokens = (size_t)atol(argv[7]);

    cis2_pin_ftz_daz();
    cis2_ftz_daz_selftest();
    printf("CIS2_VERIFY3_INT8 selftest=PASS ftz_daz=pinned\n");

    uint8_t weights_sha[32], config_sha[32], tokenizer_sha[32];
    cis2_sha256_file(weights_path, weights_sha);
    cis2_sha256_file(config_path, config_sha);
    cis2_sha256_file(tokenizer_path, tokenizer_sha);

    char *config_text = read_whole_file_or_die(config_path);
    char *tokenizer_text = read_whole_file_or_die(tokenizer_path);

    cis2_tokenizer *tok = cis2_tokenizer_load(tokenizer_text);
    uint32_t *prompt_ids = NULL;
    size_t n_prompt = 0;
    cis2_tokenizer_encode(tok, "Once upon a time", &prompt_ids, &n_prompt);

    cis2_model *m = cis2_model_load(weights_path, config_text);
    cis2_qmodel *qm = cis2_qmodel_build(m);

    /* fp32 tensor byte count (sum of all weight tensors as widened f32,
     * i.e. what the fp32 reference path holds resident) */
    size_t hidden = m->cfg.hidden_size, inter = m->cfg.intermediate_size;
    size_t nkv_dim = m->cfg.num_key_value_heads * m->head_dim;
    size_t vocab = m->cfg.vocab_size;
    size_t fp32_bytes = (vocab * hidden) * sizeof(float); /* embed_tokens */
    if (!m->cfg.tie_word_embeddings && m->lm_head) fp32_bytes += (vocab * hidden) * sizeof(float);
    fp32_bytes += m->cfg.num_hidden_layers * (
        hidden * hidden * 2 /* q,o */ +
        nkv_dim * hidden * 2 /* k,v */ +
        inter * hidden * 2 /* gate,up */ +
        hidden * inter /* down */
    ) * sizeof(float);
    printf("CIS2_VERIFY3_INT8 fp32_weight_bytes=%zu int8_weight_bytes=%zu ratio=%.4f\n",
           fp32_bytes, qm->n_bytes_quantized, (double)qm->n_bytes_quantized / (double)fp32_bytes);

    /* determinism: two in-process runs of the int8 witness chain must match */
    cis2_qrun_result r1, r2;
    cis2_qrun_decode(qm, prompt_ids, n_prompt, 16, weights_sha, tokenizer_sha, config_sha, &r1);
    cis2_qrun_decode(qm, prompt_ids, n_prompt, 16, weights_sha, tokenizer_sha, config_sha, &r2);
    int det_ok = (memcmp(r1.witness_digest, r2.witness_digest, 32) == 0) &&
                 (memcmp(r1.argmax_digest, r2.argmax_digest, 32) == 0) &&
                 (memcmp(r1.generated_ids, r2.generated_ids, 16 * sizeof(uint32_t)) == 0);
    char witness_hex[65], argmax_hex[65];
    hex_encode(r1.witness_digest, 32, witness_hex);
    hex_encode(r1.argmax_digest, 32, argmax_hex);
    printf("CIS2_VERIFY3_INT8 determinism_in_process=%s\n", det_ok ? "PASS" : "FAIL");
    printf("CIS2_VERIFY3_INT8 digest=%s dtype=int8\n", witness_hex);
    printf("CIS2_VERIFY3_INT8 argmax_digest=%s\n", argmax_hex);
    printf("CIS2_VERIFY3_INT8 generated_token_ids=");
    for (size_t i = 0; i < r1.n_gen; i++) printf("%s%u", i ? "," : "", r1.generated_ids[i]);
    printf("\n");

    /* tok/s: time a fresh 32-token int8 greedy decode */
    {
        double t0 = now_sec();
        cis2_qrun_result rt;
        cis2_qrun_decode(qm, prompt_ids, n_prompt, 32, weights_sha, tokenizer_sha, config_sha, &rt);
        double t1 = now_sec();
        double toks_per_sec = 32.0 / (t1 - t0);
        printf("CIS2_VERIFY3_INT8 timing_gen_tokens=32 seconds=%.4f tok_per_s=%.4f\n", t1 - t0, toks_per_sec);
        free(rt.generated_ids);
    }

    free(r1.generated_ids);
    free(r2.generated_ids);

    /* fp32 baseline tok/s for comparison, same 32-token decode via the
     * qm-driven forward_generic path with use_int8=0 (fp32 weights) --
     * reuses cis2_qrun_decode's machinery is int8-only, so time the
     * existing fp32 cis2_run_decode from model.c instead for a fair,
     * unmodified fp32 baseline. */
    {
        cis2_run_result rf1;
        double t0 = now_sec();
        cis2_run_decode(m, prompt_ids, n_prompt, 32, weights_sha, tokenizer_sha, config_sha, &rf1);
        double t1 = now_sec();
        printf("CIS2_VERIFY3_INT8 fp32_timing_gen_tokens=32 seconds=%.4f tok_per_s=%.4f\n",
               t1 - t0, 32.0 / (t1 - t0));
        free(rf1.generated_ids);
    }

    /* perplexity / top-1 agreement on a fixed public-text slice */
    {
        char *raw = read_whole_file_or_die(eval_text_path);
        long total_len = (long)strlen(raw);
        if (eval_offset < 0) eval_offset = 0;
        if (eval_offset > total_len) eval_offset = total_len;
        if (eval_len < 0 || eval_offset + eval_len > total_len) eval_len = total_len - eval_offset;
        char *slice = malloc((size_t)eval_len + 1);
        memcpy(slice, raw + eval_offset, (size_t)eval_len);
        slice[eval_len] = '\0';
        free(raw);

        uint32_t *eval_ids = NULL;
        size_t n_eval = 0;
        cis2_tokenizer_encode(tok, slice, &eval_ids, &n_eval);
        free(slice);

        if (n_eval > max_eval_tokens) n_eval = max_eval_tokens;
        printf("CIS2_VERIFY3_INT8 eval_tokens_used=%zu eval_tokens_requested_max=%zu\n", n_eval, max_eval_tokens);

        double nll_fp32 = 0.0, nll_int8 = 0.0;
        size_t top1_fp32 = 0, top1_int8 = 0;
        size_t n_pred = n_eval > 0 ? n_eval - 1 : 0;
        uint32_t *pred_fp32 = malloc((n_pred ? n_pred : 1) * sizeof(uint32_t));
        uint32_t *pred_int8 = malloc((n_pred ? n_pred : 1) * sizeof(uint32_t));
        double t0 = now_sec();
        cis2_eval_teacher_forced(m, qm, 0, eval_ids, n_eval, &nll_fp32, &top1_fp32, pred_fp32);
        double t1 = now_sec();
        cis2_eval_teacher_forced(m, qm, 1, eval_ids, n_eval, &nll_int8, &top1_int8, pred_int8);
        double t2 = now_sec();

        double mean_nll_fp32 = n_pred ? nll_fp32 / (double)n_pred : 0.0;
        double mean_nll_int8 = n_pred ? nll_int8 / (double)n_pred : 0.0;
        double ppl_fp32 = exp(mean_nll_fp32);
        double ppl_int8 = exp(mean_nll_int8);

        size_t agree = 0;
        for (size_t i = 0; i < n_pred; i++) if (pred_fp32[i] == pred_int8[i]) agree++;

        printf("CIS2_VERIFY3_INT8 eval_fp32_seconds=%.4f eval_int8_seconds=%.4f\n", t1 - t0, t2 - t1);
        printf("CIS2_VERIFY3_INT8 fp32_mean_nll=%.6f fp32_ppl=%.6f fp32_top1_vs_groundtruth=%zu/%zu (%.4f%%)\n",
               mean_nll_fp32, ppl_fp32, top1_fp32, n_pred, n_pred ? 100.0 * top1_fp32 / n_pred : 0.0);
        printf("CIS2_VERIFY3_INT8 int8_mean_nll=%.6f int8_ppl=%.6f int8_top1_vs_groundtruth=%zu/%zu (%.4f%%)\n",
               mean_nll_int8, ppl_int8, top1_int8, n_pred, n_pred ? 100.0 * top1_int8 / n_pred : 0.0);
        printf("CIS2_VERIFY3_INT8 int8_vs_fp32_top1_agreement=%zu/%zu (%.4f%%)\n",
               agree, n_pred, n_pred ? 100.0 * agree / n_pred : 0.0);

        free(pred_fp32);
        free(pred_int8);
        free(eval_ids);
    }

    cis2_qmodel_free(qm);
    cis2_model_free(m);
    cis2_tokenizer_free(tok);
    free(prompt_ids);
    free(tokenizer_text);
    free(config_text);
    return 0;
}
