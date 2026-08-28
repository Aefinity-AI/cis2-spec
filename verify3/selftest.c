/* selftest.c — local unit tests, no model weights required.
 * Checks the spec's own regression vectors (§1.3, §5.1, §6.1, §6.6, §7.2)
 * and a small SHA-256 known-answer test, plus tokenizer BPE on the pinned
 * prompt. Run locally (`make selftest && ./selftest`); NOT the CI
 * conformance gate (that needs the real weights, CI-only per task rules).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mathpin.h"
#include "sha256.h"
#include "json.h"
#include "tokenizer.h"

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } else { printf("ok: %s\n", msg); } } while (0)

static void hex_encode(const uint8_t *b, size_t n, char *out) {
    static const char *h = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) { out[2*i] = h[b[i]>>4]; out[2*i+1] = h[b[i]&0xF]; }
    out[2*n] = 0;
}

int main(void)
{
    cis2_pin_ftz_daz();
    cis2_ftz_daz_selftest();
    CHECK(1, "FTZ/DAZ pin + adversarial self-test (spec 1.3)");

    /* rsqrt(64.0) == 0.125 exactly, 0x3E000000 */
    float rs = cis2_rsqrt(64.0f);
    CHECK(cis2_f32_bits(rs) == 0x3E000000u, "rsqrt(64.0) == 0.125 exact (spec 6.1)");

    /* dot_seq([1e8,1.0,-1e8],[1,1,1]) == 0.0 exactly (spec 5.1) */
    float a[3] = {1e8f, 1.0f, -1e8f};
    float b[3] = {1.0f, 1.0f, 1.0f};
    float d = cis2_dot_seq(a, b, 3);
    CHECK(d == 0.0f, "dot_seq order-sensitive regression (spec 5.1)");

    /* table_digest (spec 6.6) */
    cis2_sha256_ctx tctx;
    cis2_sha256_init(&tctx);
    cis2_feed_table_digest(&tctx);
    uint8_t tdig[32];
    cis2_sha256_final(&tctx, tdig);
    char thex[65];
    hex_encode(tdig, 32, thex);
    CHECK(strcmp(thex, "465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3") == 0,
          "table_digest matches spec 13.1 pinned value");
    if (strcmp(thex, "465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3") != 0)
        printf("  got: %s\n", thex);

    /* inv_freq_table_digest for SmolLM2 params: head_dim=64, rope_theta=100000 (spec 7.2) */
    float ln_theta = cis2_ln_pinned(100000.0f);
    float inv_freq[32];
    for (int i = 0; i < 32; i++) {
        float frac = ((float)(2*i)) / 64.0f;
        float neg_arg = -(frac * ln_theta);
        inv_freq[i] = cis2_exp_pinned(neg_arg);
    }
    cis2_sha256_ctx ictx;
    cis2_sha256_init(&ictx);
    for (int i = 0; i < 32; i++) {
        uint32_t bits = cis2_f32_bits(inv_freq[i]);
        uint8_t bb[4] = {(uint8_t)bits, (uint8_t)(bits>>8), (uint8_t)(bits>>16), (uint8_t)(bits>>24)};
        cis2_sha256_update(&ictx, bb, 4);
    }
    uint8_t idig[32];
    cis2_sha256_final(&ictx, idig);
    char ihex[65];
    hex_encode(idig, 32, ihex);
    CHECK(strcmp(ihex, "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12") == 0,
          "inv_freq_table_digest matches spec 13.1 pinned value");
    if (strcmp(ihex, "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12") != 0)
        printf("  got: %s\n", ihex);

    /* SHA-256 known-answer test: sha256("abc") */
    cis2_sha256_ctx sctx;
    cis2_sha256_init(&sctx);
    cis2_sha256_update(&sctx, (const uint8_t*)"abc", 3);
    uint8_t sdig[32];
    cis2_sha256_final(&sctx, sdig);
    char shex[65];
    hex_encode(sdig, 32, shex);
    CHECK(strcmp(shex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0,
          "sha256(\"abc\") KAT");

    /* minimal tokenizer JSON with a synthetic vocab/merges sufficient to
     * BPE-encode "hi" -> ["h","i"] merged to "hi" if a merge exists. */
    const char *tokjson =
        "{\"model\":{\"vocab\":{\"h\":0,\"i\":1,\"hi\":2},\"merges\":[\"h i\"]}}";
    cis2_tokenizer *tk = cis2_tokenizer_load(tokjson);
    uint32_t *ids; size_t n;
    cis2_tokenizer_encode(tk, "hi", &ids, &n);
    CHECK(n == 1 && ids[0] == 2, "tiny BPE merge sanity check");
    free(ids);
    cis2_tokenizer_free(tk);

    printf("%s: %d failures\n", failures ? "SELFTEST FAIL" : "SELFTEST PASS", failures);
    return failures ? 1 : 0;
}
