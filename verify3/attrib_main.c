/* attrib_main.c — xb-2 driver: attention-rollout attribution for an
 * arbitrary prompt, on top of the same bit-exact fp32 CIS-2 forward pass
 * used by cis2_verify3 (main.c / model.c). Not part of the normative
 * CIS-2 witness chain; a separate, additive side-channel.
 *
 * Usage: cis2_attrib <weights_dir> "<prompt text>"
 *   weights_dir must contain model.safetensors, config.json, tokenizer.json
 *   (artifact hashes are checked against the same pinned SmolLM2-135M
 *   digests as cis2_verify3, so this only runs against the exact model
 *   verify3 was built for).
 *
 * Output (stable, greppable):
 *   CIS2_ATTRIB prompt_sha256=<hex>
 *   CIS2_ATTRIB prompt_tok_count=<n>
 *   CIS2_ATTRIB rollout_digest=<hex>
 *   CIS2_ATTRIB token[<i>]=<id> score=<fp32 decimal>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mathpin.h"
#include "model.h"
#include "tokenizer.h"
#include "sha256.h"

static const uint8_t PINNED_WEIGHTS_SHA256[32] = {
    0x80,0x52,0x1b,0x40,0x28,0x1d,0x6c,0xe7,0x4e,0x35,0xc9,0x28,0x2c,0x22,0x53,0x9e,
    0x75,0xaa,0x0a,0xc8,0x57,0x88,0x92,0xb2,0xa5,0x99,0x55,0xef,0x78,0xd5,0x5d,0xa1
};

static void hex_encode(const uint8_t *bytes, size_t n, char *out)
{
    static const char *hexd = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = hexd[(bytes[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[bytes[i] & 0xF];
    }
    out[2 * n] = '\0';
}

static char *read_whole_file_or_die(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "CIS2_ATTRIB FATAL: cannot open %s\n", path); exit(90); }
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
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weights_dir> \"<prompt text>\"\n", argv[0]);
        return 2;
    }
    const char *weights_dir = argv[1];
    const char *prompt = argv[2];

    char weights_path[4096], config_path[4096], tokenizer_path[4096];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", weights_dir);
    snprintf(config_path, sizeof(config_path), "%s/config.json", weights_dir);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", weights_dir);

    cis2_pin_ftz_daz();
    cis2_ftz_daz_selftest();

    uint8_t weights_sha[32];
    if (cis2_sha256_file(weights_path, weights_sha) != 0) {
        fprintf(stderr, "CIS2_ATTRIB FATAL: cannot hash %s\n", weights_path);
        return 90;
    }
    if (memcmp(weights_sha, PINNED_WEIGHTS_SHA256, 32) != 0) {
        char got[65]; hex_encode(weights_sha, 32, got);
        fprintf(stderr, "CIS2_ATTRIB FATAL: model.safetensors sha256 mismatch, got=%s\n", got);
        return 91;
    }

    char *config_text = read_whole_file_or_die(config_path);
    char *tokenizer_text = read_whole_file_or_die(tokenizer_path);

    cis2_tokenizer *tok = cis2_tokenizer_load(tokenizer_text);
    uint32_t *prompt_ids = NULL;
    size_t n_prompt = 0;
    cis2_tokenizer_encode(tok, prompt, &prompt_ids, &n_prompt);
    cis2_tokenizer_free(tok);
    free(tokenizer_text);

    if (n_prompt < 1) {
        fprintf(stderr, "CIS2_ATTRIB FATAL: empty tokenization\n");
        return 92;
    }

    uint8_t prompt_sha[32];
    cis2_sha256_ctx pctx;
    cis2_sha256_init(&pctx);
    cis2_sha256_update(&pctx, (const uint8_t *)prompt, strlen(prompt));
    cis2_sha256_final(&pctx, prompt_sha);
    char prompt_sha_hex[65];
    hex_encode(prompt_sha, 32, prompt_sha_hex);

    cis2_model *m = cis2_model_load(weights_path, config_text);
    free(config_text);

    cis2_attrib_result r1, r2;
    cis2_run_attribution(m, prompt_ids, n_prompt, &r1);
    cis2_run_attribution(m, prompt_ids, n_prompt, &r2);

    int det_ok = (memcmp(r1.rollout_digest, r2.rollout_digest, 32) == 0);
    if (!det_ok) {
        fprintf(stderr, "CIS2_ATTRIB FATAL: two-run determinism check failed\n");
        return 93;
    }

    char digest_hex[65];
    hex_encode(r1.rollout_digest, 32, digest_hex);

    printf("CIS2_ATTRIB prompt_sha256=%s\n", prompt_sha_hex);
    printf("CIS2_ATTRIB prompt_tok_count=%zu\n", n_prompt);
    printf("CIS2_ATTRIB rollout_digest=%s\n", digest_hex);
    for (size_t i = 0; i < n_prompt; i++) {
        printf("CIS2_ATTRIB token[%zu]=%u score=%.9f\n", i, prompt_ids[i], (double)r1.rollout[i]);
    }

    cis2_attrib_free(&r1);
    cis2_attrib_free(&r2);
    free(prompt_ids);
    cis2_model_free(m);
    return 0;
}
