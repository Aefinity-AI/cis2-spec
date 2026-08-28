/* sha256.h
 * Vendored, adapted from Brad Conte's crypto-algorithms (public domain,
 * https://github.com/B-Con/crypto-algorithms). No rights reserved by the
 * original author. Adapted naming only for this codebase (verify3/).
 */
#ifndef CIS2_V3_SHA256_H
#define CIS2_V3_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define CIS2_SHA256_BLOCK_SIZE 32

typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} cis2_sha256_ctx;

void cis2_sha256_init(cis2_sha256_ctx *ctx);
void cis2_sha256_update(cis2_sha256_ctx *ctx, const uint8_t data[], size_t len);
void cis2_sha256_final(cis2_sha256_ctx *ctx, uint8_t hash[CIS2_SHA256_BLOCK_SIZE]);

#endif
