/* main.c — CIS-2 v0.2 clean-room verify3 (C11) driver. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mathpin.h"
#include "model.h"
#include "tokenizer.h"

static const uint8_t PINNED_WEIGHTS_SHA256[32] = {
    0x80,0x52,0x1b,0x40,0x28,0x1d,0x6c,0xe7,0x4e,0x35,0xc9,0x28,0x2c,0x22,0x53,0x9e,
    0x75,0xaa,0x0a,0xc8,0x57,0x88,0x92,0xb2,0xa5,0x99,0x55,0xef,0x78,0xd5,0x5d,0xa1
};
static const uint8_t PINNED_CONFIG_SHA256[32] = {
    0x1d,0x55,0x6e,0xab,0x73,0xb6,0x9c,0x7f,0x11,0xf6,0x4c,0x55,0x7a,0x2f,0x9c,0x6f,
    0x44,0x0b,0xd4,0xc6,0xb8,0x9b,0xb2,0x58,0x4a,0x6b,0x49,0x8c,0x92,0x60,0x38,0x43
};
static const uint8_t PINNED_TOKENIZER_SHA256[32] = {
    0x9c,0xa9,0xac,0xdd,0xb6,0x52,0x5a,0x19,0x4e,0xc8,0xac,0x7a,0x87,0xf2,0x4f,0xbb,
    0xa7,0x23,0x2a,0x9a,0x15,0xff,0xa1,0xaf,0x0c,0x12,0x24,0xfc,0xd8,0x88,0xe4,0x7c
};

static const uint32_t PINNED_PROMPT_IDS[4] = {6403, 1980, 253, 655};
static const uint32_t PINNED_GENERATED[16] = {
    28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351
};
static const char *PINNED_TABLE_DIGEST_HEX =
    "23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d";
static const char *PINNED_INV_FREQ_DIGEST_HEX =
    "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12";
static const char *PINNED_ARGMAX_DIGEST_HEX =
    "0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff";
static const char *PINNED_WITNESS_DIGEST_HEX =
    "d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df";

static void hex_encode(const uint8_t *bytes, size_t n, char *out /* 2n+1 */)
{
    static const char *hexd = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = hexd[(bytes[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[bytes[i] & 0xF];
    }
    out[2 * n] = '\0';
}

static int hex_matches(const uint8_t *bytes, size_t n, const char *expected_hex)
{
    char buf[256];
    hex_encode(bytes, n, buf);
    return strcmp(buf, expected_hex) == 0;
}

static char *read_whole_file_or_die(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: cannot open %s\n", path);
        exit(90);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    const char *weights_path = argc > 1 ? argv[1] : "weights/model.safetensors";
    const char *config_path = argc > 2 ? argv[2] : "weights/config.json";
    const char *tokenizer_path = argc > 3 ? argv[3] : "weights/tokenizer.json";

    cis2_pin_ftz_daz();
    cis2_ftz_daz_selftest();
    printf("CIS2_VERIFY3 selftest=PASS ftz_daz=pinned\n");

    /* E15k debug-only: per-layer state dump side-channel, mirrors the
     * Rust reference's CIS2_DUMP_LAYERS env var. Not part of CIS2_REF. */
    cis2_set_dump_layers(getenv("CIS2_DUMP_LAYERS") != NULL);

    uint8_t weights_sha[32], config_sha[32], tokenizer_sha[32];
    if (cis2_sha256_file(weights_path, weights_sha) != 0) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: cannot hash %s\n", weights_path);
        return 90;
    }
    if (cis2_sha256_file(config_path, config_sha) != 0) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: cannot hash %s\n", config_path);
        return 90;
    }
    if (cis2_sha256_file(tokenizer_path, tokenizer_sha) != 0) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: cannot hash %s\n", tokenizer_path);
        return 90;
    }

    if (memcmp(weights_sha, PINNED_WEIGHTS_SHA256, 32) != 0) {
        char got[65]; hex_encode(weights_sha, 32, got);
        fprintf(stderr, "CIS2_VERIFY3 FATAL: model.safetensors sha256 mismatch, got=%s\n", got);
        return 91;
    }
    if (memcmp(config_sha, PINNED_CONFIG_SHA256, 32) != 0) {
        char got[65]; hex_encode(config_sha, 32, got);
        fprintf(stderr, "CIS2_VERIFY3 FATAL: config.json sha256 mismatch, got=%s\n", got);
        return 91;
    }
    if (memcmp(tokenizer_sha, PINNED_TOKENIZER_SHA256, 32) != 0) {
        char got[65]; hex_encode(tokenizer_sha, 32, got);
        fprintf(stderr, "CIS2_VERIFY3 FATAL: tokenizer.json sha256 mismatch, got=%s\n", got);
        return 91;
    }
    printf("CIS2_VERIFY3 artifact_hashes=PASS\n");

    char *config_text = read_whole_file_or_die(config_path);
    char *tokenizer_text = read_whole_file_or_die(tokenizer_path);

    cis2_tokenizer *tok = cis2_tokenizer_load(tokenizer_text);
    uint32_t *prompt_ids = NULL;
    size_t n_prompt = 0;
    cis2_tokenizer_encode(tok, "Once upon a time", &prompt_ids, &n_prompt);
    cis2_tokenizer_free(tok);
    free(tokenizer_text);

    if (n_prompt != 4 || memcmp(prompt_ids, PINNED_PROMPT_IDS, 4 * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: prompt tokenization mismatch (n=%zu):", n_prompt);
        for (size_t i = 0; i < n_prompt; i++) fprintf(stderr, " %u", prompt_ids[i]);
        fprintf(stderr, " (expected 6403 1980 253 655)\n");
        return 92;
    }

    cis2_model *m = cis2_model_load(weights_path, config_text);
    free(config_text);

    cis2_run_result r1, r2;
    cis2_run_decode(m, prompt_ids, n_prompt, 16, weights_sha, tokenizer_sha, config_sha, &r1);
    cis2_run_decode(m, prompt_ids, n_prompt, 16, weights_sha, tokenizer_sha, config_sha, &r2);

    /* §12.4 same-process two-run determinism check */
    int det_ok = 1;
    if (memcmp(r1.witness_digest, r2.witness_digest, 32) != 0) det_ok = 0;
    if (memcmp(r1.argmax_digest, r2.argmax_digest, 32) != 0) det_ok = 0;
    if (memcmp(r1.generated_ids, r2.generated_ids, 16 * sizeof(uint32_t)) != 0) det_ok = 0;
    if (!det_ok) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: two-run determinism check failed (spec §12.4)\n");
        return 93;
    }
    printf("CIS2_VERIFY3 determinism=PASS\n");

    char witness_hex[65], argmax_hex[65], table_hex[65], invfreq_hex[65];
    hex_encode(r1.witness_digest, 32, witness_hex);
    hex_encode(r1.argmax_digest, 32, argmax_hex);
    hex_encode(r1.table_digest, 32, table_hex);
    hex_encode(r1.inv_freq_table_digest, 32, invfreq_hex);

    printf("CIS2_VERIFY3 digest=%s prompt_idx=0 prompt_toks=4 gen_toks=16 dtype=fp32\n", witness_hex);
    printf("CIS2_VERIFY3 argmax_digest=%s\n", argmax_hex);
    printf("CIS2_VERIFY3 table_digest=%s\n", table_hex);
    printf("CIS2_VERIFY3 inv_freq_table_digest=%s\n", invfreq_hex);

    printf("CIS2_VERIFY3 generated_token_ids=");
    for (size_t i = 0; i < r1.n_gen; i++) printf("%s%u", i ? "," : "", r1.generated_ids[i]);
    printf("\n");

    int conform_ok = 1;
    if (!hex_matches(r1.witness_digest, 32, PINNED_WITNESS_DIGEST_HEX)) {
        fprintf(stderr, "CIS2_VERIFY3 MISMATCH: witness digest expected=%s got=%s\n",
                PINNED_WITNESS_DIGEST_HEX, witness_hex);
        conform_ok = 0;
    }
    if (!hex_matches(r1.argmax_digest, 32, PINNED_ARGMAX_DIGEST_HEX)) {
        fprintf(stderr, "CIS2_VERIFY3 MISMATCH: argmax digest expected=%s got=%s\n",
                PINNED_ARGMAX_DIGEST_HEX, argmax_hex);
        conform_ok = 0;
    }
    if (!hex_matches(r1.table_digest, 32, PINNED_TABLE_DIGEST_HEX)) {
        fprintf(stderr, "CIS2_VERIFY3 MISMATCH: table digest expected=%s got=%s\n",
                PINNED_TABLE_DIGEST_HEX, table_hex);
        conform_ok = 0;
    }
    if (!hex_matches(r1.inv_freq_table_digest, 32, PINNED_INV_FREQ_DIGEST_HEX)) {
        fprintf(stderr, "CIS2_VERIFY3 MISMATCH: inv_freq table digest expected=%s got=%s\n",
                PINNED_INV_FREQ_DIGEST_HEX, invfreq_hex);
        conform_ok = 0;
    }
    if (r1.n_gen != 16 || memcmp(r1.generated_ids, PINNED_GENERATED, 16 * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "CIS2_VERIFY3 MISMATCH: generated_token_ids differ from §13.1 pinned vector\n");
        conform_ok = 0;
    }

    free(r1.generated_ids);
    free(r2.generated_ids);
    free(prompt_ids);
    cis2_model_free(m);

    if (!conform_ok) {
        printf("CIS2_VERIFY3 conformance=FAIL\n");
        return 1;
    }
    printf("CIS2_VERIFY3 conformance=PASS\n");
    return 0;
}
