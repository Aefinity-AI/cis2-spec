/* main_spec.c — driver for QUEUE item spec-1 (EXACT SPECULATIVE DECODING).
 * Runs plain fp32 KV-cache greedy decode vs int8-draft/fp32-target
 * speculative decode on a fixed 5-prompt set, 64 tokens each, and checks
 * that generated token ids + argmax digest are byte-identical between
 * the two. Also reports acceptance rate per prompt and tok/s before/after.
 *
 * Usage: cis2_verify3_spec weights.safetensors config.json tokenizer.json
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mathpin.h"
#include "model.h"
#include "model_int8.h"
#include "model_spec.h"
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

static const char *PROMPTS[5] = {
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog and then",
    "In a shocking turn of events, scientists announced today that",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The history of computing spans many decades of steady progress, beginning with mechanical calculators and evolving through vacuum tubes, transistors, integrated circuits, and eventually the modern microprocessor."
};

#define N_GEN 64
#define SPEC_K 4

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.safetensors config.json tokenizer.json\n", argv[0]);
        return 2;
    }
    const char *weights_path = argv[1];
    const char *config_path = argv[2];
    const char *tokenizer_path = argv[3];

    cis2_pin_ftz_daz();
    cis2_ftz_daz_selftest();
    printf("CIS2_VERIFY3_SPEC selftest=PASS ftz_daz=pinned\n");

    uint8_t weights_sha[32], config_sha[32], tokenizer_sha[32];
    cis2_sha256_file(weights_path, weights_sha);
    cis2_sha256_file(config_path, config_sha);
    cis2_sha256_file(tokenizer_path, tokenizer_sha);

    char *config_text = read_whole_file_or_die(config_path);
    char *tokenizer_text = read_whole_file_or_die(tokenizer_path);
    cis2_tokenizer *tok = cis2_tokenizer_load(tokenizer_text);

    cis2_model *m = cis2_model_load(weights_path, config_text);
    cis2_qmodel *qm = cis2_qmodel_build(m);

    int all_pass = 1;
    double total_plain_sec = 0.0, total_spec_sec = 0.0;
    size_t total_gen_tokens = 0;

    for (int p = 0; p < 5; p++) {
        uint32_t *prompt_ids = NULL;
        size_t n_prompt = 0;
        cis2_tokenizer_encode(tok, PROMPTS[p], &prompt_ids, &n_prompt);

        double t0 = now_sec();
        cis2_run_result plain;
        cis2_run_decode(m, prompt_ids, n_prompt, N_GEN, weights_sha, tokenizer_sha, config_sha, &plain);
        double t1 = now_sec();
        double plain_sec = t1 - t0;

        double t2 = now_sec();
        cis2_spec_run_result spec;
        cis2_spec_run_decode(m, qm, prompt_ids, n_prompt, N_GEN, SPEC_K,
                              weights_sha, tokenizer_sha, config_sha, &spec);
        double t3 = now_sec();
        double spec_sec = t3 - t2;

        total_plain_sec += plain_sec;
        total_spec_sec += spec_sec;
        total_gen_tokens += N_GEN;

        int ids_match = (plain.n_gen == spec.n_gen) &&
                         (memcmp(plain.generated_ids, spec.generated_ids, N_GEN * sizeof(uint32_t)) == 0);
        int argmax_match = memcmp(plain.argmax_digest, spec.argmax_digest, 32) == 0;
        int witness_match = memcmp(plain.witness_digest, spec.witness_digest, 32) == 0;
        int pass = ids_match && argmax_match;
        if (!pass) all_pass = 0;

        char plain_argmax_hex[65], spec_argmax_hex[65];
        hex_encode(plain.argmax_digest, 32, plain_argmax_hex);
        hex_encode(spec.argmax_digest, 32, spec_argmax_hex);

        size_t drafted = spec.total_draft_proposed;
        size_t accepted = spec.total_draft_accepted;
        double accept_rate = drafted ? (100.0 * (double)accepted / (double)drafted) : 0.0;

        printf("CIS2_VERIFY3_SPEC prompt_idx=%d n_prompt=%zu\n", p, n_prompt);
        printf("CIS2_VERIFY3_SPEC prompt_idx=%d ids_match=%s argmax_digest_match=%s witness_digest_match=%s\n",
               p, ids_match ? "PASS" : "FAIL", argmax_match ? "PASS" : "FAIL", witness_match ? "PASS" : "FAIL");
        printf("CIS2_VERIFY3_SPEC prompt_idx=%d plain_argmax_digest=%s spec_argmax_digest=%s\n",
               p, plain_argmax_hex, spec_argmax_hex);
        printf("CIS2_VERIFY3_SPEC prompt_idx=%d rounds=%zu draft_proposed=%zu draft_accepted=%zu accept_rate=%.2f%% bonus_accepted=%zu\n",
               p, spec.n_rounds, drafted, accepted, accept_rate, spec.total_bonus_accepted);
        printf("CIS2_VERIFY3_SPEC prompt_idx=%d plain_seconds=%.4f plain_tok_per_s=%.4f spec_seconds=%.4f spec_tok_per_s=%.4f speedup=%.3fx\n",
               p, plain_sec, N_GEN / plain_sec, spec_sec, N_GEN / spec_sec, plain_sec / spec_sec);
        if (!ids_match) {
            printf("CIS2_VERIFY3_SPEC prompt_idx=%d plain_ids=", p);
            for (size_t i = 0; i < plain.n_gen; i++) printf("%s%u", i ? "," : "", plain.generated_ids[i]);
            printf("\n");
            printf("CIS2_VERIFY3_SPEC prompt_idx=%d spec_ids=", p);
            for (size_t i = 0; i < spec.n_gen; i++) printf("%s%u", i ? "," : "", spec.generated_ids[i]);
            printf("\n");
        }

        free(plain.generated_ids);
        free(spec.generated_ids);
        free(prompt_ids);
    }

    printf("CIS2_VERIFY3_SPEC total_plain_seconds=%.4f total_spec_seconds=%.4f total_gen_tokens=%zu overall_speedup=%.3fx\n",
           total_plain_sec, total_spec_sec, total_gen_tokens, total_plain_sec / total_spec_sec);
    printf("CIS2_VERIFY3_SPEC overall=%s\n", all_pass ? "PASS" : "FAIL");

    cis2_qmodel_free(qm);
    cis2_model_free(m);
    cis2_tokenizer_free(tok);
    free(tokenizer_text);
    free(config_text);
    return all_pass ? 0 : 1;
}
