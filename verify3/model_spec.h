/* model_spec.h — spec-1: exact/lossless speculative decoding.
 *
 * Draft: the int8 path (model_int8.c, QUEUE item small-1) proposes k
 * tokens ahead, greedily, one at a time (cis2_qdraft_next).
 * Target: the fp32 KV-cache path (model.c, QUEUE item fast-1) verifies
 * each proposed token greedily and accepts/rejects per the standard
 * speculative-decoding algorithm (Leviathan et al. 2023, "Fast Inference
 * from Transformers via Speculative Decoding"; Chen et al. 2023,
 * "Accelerating Large Language Model Decoding with Speculative
 * Sampling") -- NOT a novel technique, just applied here. Because this is
 * the *greedy* (not sampled) specialization, the accept rule reduces to
 * "accept iff draft's proposed token equals target's own greedy argmax at
 * that position"; on any rejection (or once k drafted tokens are all
 * accepted) the target's own greedy choice is what gets committed --
 * this is lossless by construction: the sequence of committed tokens is
 * always exactly the target model's greedy continuation, identical to
 * plain (non-speculative) fp32 KV-cache decoding, token for token. The
 * only purpose of the draft is to let the target verify multiple
 * candidate positions in a single batched pass (model.c
 * cis2_process_positions_batch) instead of a strictly serial per-token
 * pass -- see that function's comment for the memory-bandwidth mechanism.
 */
#ifndef CIS2_V3_MODEL_SPEC_H
#define CIS2_V3_MODEL_SPEC_H

#include <stdint.h>
#include <stddef.h>
#include "model.h"
#include "model_int8.h"

typedef struct {
    uint32_t *generated_ids; /* [n_gen] */
    size_t n_gen;
    uint8_t witness_digest[32];
    uint8_t argmax_digest[32];

    /* acceptance stats */
    size_t n_rounds;
    size_t total_draft_proposed;  /* sum of k_eff over all rounds */
    size_t total_draft_accepted;  /* draft tokens that matched target's argmax */
    size_t total_bonus_accepted;  /* "free" extra tokens from fully-accepted rounds */
} cis2_spec_run_result;

/* k = number of tokens the draft proposes per round (spec-1 uses k=4).
 * Same witness-chain / argmax-chain construction and byte layout as
 * cis2_run_decode (model.c) -- every committed token's full logits vector
 * (as computed by the target) is fed into the witness digest in
 * generation order, and every committed token id is fed into both
 * digests, identically to the non-speculative path, so a correct
 * implementation produces byte-identical witness_digest/argmax_digest/
 * generated_ids to cis2_run_decode given the same inputs. */
void cis2_spec_run_decode(const cis2_model *m, const cis2_qmodel *qm,
                           const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen, size_t k,
                           const uint8_t weights_sha256[32],
                           const uint8_t tokenizer_sha256[32],
                           const uint8_t config_sha256[32],
                           cis2_spec_run_result *out);

#endif
