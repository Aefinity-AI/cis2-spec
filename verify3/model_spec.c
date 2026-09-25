/* model_spec.c — see model_spec.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_spec.h"
#include "mathpin.h"
#include "sha256.h"

static void feed_u32_le(cis2_sha256_ctx *ctx, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
                     (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF) };
    cis2_sha256_update(ctx, b, 4);
}

static void feed_logits(cis2_sha256_ctx *ctx, const float *logits, size_t vocab)
{
    for (size_t i = 0; i < vocab; i++) {
        uint32_t bits = cis2_f32_bits(logits[i]);
        uint8_t b[4] = { (uint8_t)(bits & 0xFF), (uint8_t)((bits >> 8) & 0xFF),
                         (uint8_t)((bits >> 16) & 0xFF), (uint8_t)((bits >> 24) & 0xFF) };
        cis2_sha256_update(ctx, b, 4);
    }
}

/* Commits one generated token (position = ntok-1 before this call, i.e.
 * this is the `n_prompt+step`-th token overall): feeds `logits` (the
 * target's logits that determined it, same as cis2_run_decode's per-step
 * witness feed) and `token` into witness+argctx, exactly matching
 * cis2_run_decode's per-step order (full logits vector, then the token
 * id, into witness; token id into argctx). */
static void commit_token(cis2_sha256_ctx *witness, cis2_sha256_ctx *argctx,
                          const float *logits, size_t vocab, uint32_t token)
{
    feed_logits(witness, logits, vocab);
    feed_u32_le(witness, token);
    feed_u32_le(argctx, token);
}

void cis2_spec_run_decode(const cis2_model *m, const cis2_qmodel *qm,
                           const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen, size_t k,
                           const uint8_t weights_sha256[32],
                           const uint8_t tokenizer_sha256[32],
                           const uint8_t config_sha256[32],
                           cis2_spec_run_result *out)
{
    /* §6.6 / §7.2 tables -- identical construction to cis2_run_decode,
     * needed only to reproduce the same witness prefix. */
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
    size_t vocab = m->cfg.vocab_size;

    size_t n_rounds = 0, total_proposed = 0, total_accepted = 0, total_bonus = 0;

    cis2_decode_kv_cache *cache = cis2_kv_cache_alloc(m, cap);
    /* spec-1: the draft (int8) model gets its own incremental KV-cache
     * (cis2_qprocess_position), same struct type/cap as the target's --
     * the cache is just float K/V arrays, dtype-independent -- so
     * proposing k tokens ahead is O(k) per round instead of O(k*ntok)
     * (see model_int8.h cis2_qprocess_position). */
    cis2_decode_kv_cache *draft_cache = cis2_kv_cache_alloc(m, cap);

    /* Prime both caches with the prompt positions (same as
     * cis2_run_decode); only the last prompt position needs logits (for
     * the target, to compare against the draft's first proposal; for the
     * draft, to seed its own greedy self-continuation). */
    float *logits_prev = NULL;       /* target's logits at position ntok-1 */
    float *draft_logits_prev = NULL; /* draft's logits at position ntok-1 */
    for (size_t p = 0; p < n_prompt; p++) {
        int need_logits = (p == n_prompt - 1);
        float *lg = cis2_process_position(m, cache, p, tokens[p], need_logits);
        if (need_logits) logits_prev = lg;
        float *dlg = cis2_qprocess_position(qm, draft_cache, p, tokens[p], need_logits);
        if (need_logits) draft_logits_prev = dlg;
    }

    size_t generated_count = 0;
    while (generated_count < n_gen) {
        size_t remaining = n_gen - generated_count;
        size_t k_eff = k < remaining ? k : remaining;
        n_rounds++;

        /* DRAFT: propose k_eff tokens, greedily, one at a time, using the
         * draft's own incremental (KV-cached) int8 forward pass --
         * exactly the draft's greedy self-continuation, O(k_eff) not
         * O(k_eff*ntok). draft_logits_prev/draft_logits[i-1] play the
         * same role for the draft's own self-comparison as
         * logits_prev/logits_batch do for the target's verification. */
        uint32_t *draft_tokens = malloc(k_eff * sizeof(uint32_t));
        float **draft_logits = malloc(k_eff * sizeof(float *));
        for (size_t i = 0; i < k_eff; i++) {
            const float *dli = (i == 0) ? draft_logits_prev : draft_logits[i - 1];
            uint32_t d = cis2_argmax(dli, vocab);
            draft_tokens[i] = d;
            draft_logits[i] = cis2_qprocess_position(qm, draft_cache, ntok + i, d, 1);
        }
        total_proposed += k_eff;

        /* TARGET: batched verification over positions [ntok, ntok+k_eff). */
        float **logits_batch = malloc(k_eff * sizeof(float *));
        for (size_t i = 0; i < k_eff; i++) logits_batch[i] = malloc(vocab * sizeof(float));
        cis2_process_positions_batch(m, cache, ntok, draft_tokens, k_eff, logits_batch);

        /* Accept/reject: t[i] is the target's greedy prediction that draft
         * token i must match. t[0] comes from logits_prev (already
         * computed, from the previous round / cache priming); t[i] for
         * i>=1 comes from logits_batch[i-1] (target's logits after
         * processing position ntok+i-1, i.e. after draft_tokens[i-1]). */
        size_t accepted = 0;
        uint32_t committed_extra_token = 0;
        const float *committed_extra_logits = NULL;
        int have_extra = 0;
        for (size_t i = 0; i < k_eff; i++) {
            const float *li = (i == 0) ? logits_prev : logits_batch[i - 1];
            uint32_t t_i = cis2_argmax(li, vocab);
            if (t_i == draft_tokens[i]) {
                /* accept: commit draft_tokens[i] using li (the logits that
                 * determined it), exactly matching cis2_run_decode's
                 * per-step witness feed order. */
                commit_token(&witness, &argctx, li, vocab, draft_tokens[i]);
                generated[generated_count++] = draft_tokens[i];
                tokens[ntok + i] = draft_tokens[i];
                accepted++;
            } else {
                /* reject: t_i (target's own choice) is committed instead,
                 * discarding draft_tokens[i..k_eff). */
                have_extra = 1;
                committed_extra_token = t_i;
                committed_extra_logits = li;
                break;
            }
        }

        if (!have_extra && accepted == k_eff && generated_count < n_gen) {
            /* all k_eff draft tokens accepted, and there's still room in
             * n_gen for a bonus token -- the batch's LAST position
             * (ntok+k_eff-1) already computed it for free. */
            uint32_t bonus = cis2_argmax(logits_batch[k_eff - 1], vocab);
            commit_token(&witness, &argctx, logits_batch[k_eff - 1], vocab, bonus);
            generated[generated_count++] = bonus;
            tokens[ntok + accepted] = bonus;
            total_bonus++;
            accepted++; /* bonus counts toward this round's committed length */
            /* need fresh logits for the NEXT round's t[0]: process the
             * bonus token's own position (single, unbatched -- the batch
             * did not compute this, its positions stopped at k_eff-1). */
            free(logits_prev);
            logits_prev = cis2_process_position(m, cache, ntok + accepted - 1, bonus, 1);
        } else if (!have_extra && accepted == k_eff) {
            /* all k_eff draft tokens accepted and n_gen is now exhausted
             * -- no room for (and no need for) a bonus token or fresh
             * logits_prev; the while loop is about to terminate. */
            free(logits_prev);
            logits_prev = NULL;
        } else {
            /* rejection path: commit the corrective token. */
            commit_token(&witness, &argctx, committed_extra_logits, vocab, committed_extra_token);
            generated[generated_count++] = committed_extra_token;
            tokens[ntok + accepted] = committed_extra_token;
            accepted++;
            /* the batch's positions >= ntok+accepted-1 used unverified
             * draft tokens and must be redone: recompute this position's
             * K/V + logits with the real (corrective) token so the cache
             * is correct going forward and we have logits_prev for the
             * next round. */
            free(logits_prev);
            logits_prev = cis2_process_position(m, cache, ntok + accepted - 1, committed_extra_token, 1);
        }
        total_accepted += (have_extra ? accepted - 1 : k_eff);

        for (size_t i = 0; i < k_eff; i++) free(logits_batch[i]);
        free(logits_batch);
        free(draft_tokens);
        for (size_t i = 0; i < k_eff; i++) free(draft_logits[i]);
        free(draft_logits);

        ntok += accepted;

        /* Re-sync the draft's own cache/logits_prev for whatever token
         * actually got committed at the round's last position (accepted
         * draft tokens before it are already correctly cached -- the
         * draft proposed those exact values itself; but the round's
         * final token is either a target correction the draft never
         * proposed, or (full-accept case) a bonus token the draft hasn't
         * seen at all -- either way the draft cache needs this one more
         * position before it can propose further). Skipped once n_gen is
         * exhausted (nothing left to draft). */
        if (generated_count < n_gen) {
            free(draft_logits_prev);
            draft_logits_prev = cis2_qprocess_position(qm, draft_cache, ntok - 1, tokens[ntok - 1], 1);
        } else {
            free(draft_logits_prev);
            draft_logits_prev = NULL;
        }
    }

    free(logits_prev);
    free(draft_logits_prev);
    cis2_kv_cache_free(cache, m);
    cis2_kv_cache_free(draft_cache, m);

    cis2_sha256_final(&witness, out->witness_digest);
    cis2_sha256_final(&argctx, out->argmax_digest);
    out->generated_ids = generated;
    out->n_gen = n_gen;
    out->n_rounds = n_rounds;
    out->total_draft_proposed = total_proposed;
    out->total_draft_accepted = total_accepted;
    out->total_bonus_accepted = total_bonus;
    free(tokens);
}
