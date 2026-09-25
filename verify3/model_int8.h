/* model_int8.h — int8 per-channel-weight / int32-accumulation forward pass
 * built on top of the fp32 cis2_model (QUEUE item small-1). Only the
 * weight-matmul reductions (q/k/v/o/gate/up/down/embed-as-lm_head) are
 * int8-quantized with int32 accumulation; RMSNorm, RoPE, softmax and the
 * attention score/value reductions are untouched fp32 (same pinned
 * primitives as verify3/mathpin.c) — this file adds a path, it does not
 * modify the existing fp32 digests (witness/argmax/table/inv_freq).
 */
#ifndef CIS2_V3_MODEL_INT8_H
#define CIS2_V3_MODEL_INT8_H

#include <stdint.h>
#include <stddef.h>
#include "model.h"
#include "quant.h"

typedef struct {
    cis2_qmat q_proj, k_proj, v_proj, o_proj;
    cis2_qmat gate_proj, up_proj, down_proj;
} cis2_qlayer;

typedef struct {
    const cis2_model *base; /* not owned */
    cis2_qmat embed; /* [vocab, hidden]; also used as lm_head when tied */
    cis2_qmat lm_head; /* only populated if !tie_word_embeddings */
    int has_separate_lm_head;
    cis2_qlayer *layers;
    size_t n_bytes_quantized; /* total bytes of q[] + scale[] arrays, for size reporting */
} cis2_qmodel;

/* Builds the quantized weight cache from an already-loaded fp32 model. */
cis2_qmodel *cis2_qmodel_build(const cis2_model *m);
void cis2_qmodel_free(cis2_qmodel *qm);

typedef struct {
    uint32_t *generated_ids;
    size_t n_gen;
    uint8_t witness_digest[32];
    uint8_t argmax_digest[32];
} cis2_qrun_result;

/* Same witness-chain construction as cis2_run_decode (model.c), but every
 * weight matmul in the forward pass runs through cis2_matvec_int8 instead
 * of cis2_matvec. table_digest / inv_freq_table_digest are fed into the
 * chain identically (they don't depend on weight dtype) but are not
 * returned separately here since they're unchanged from the fp32 path. */
void cis2_qrun_decode(const cis2_qmodel *qm,
                       const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen,
                       const uint8_t weights_sha256[32],
                       const uint8_t tokenizer_sha256[32],
                       const uint8_t config_sha256[32],
                       cis2_qrun_result *out);

/* Teacher-forced single pass over tokens[0..ntok-1] (no re-decoding):
 * for each position p in [0, ntok-2], compares argmax(logits at p) against
 * tokens[p+1] (top-1 agreement) and accumulates cross-entropy of the
 * actual next token under the model's softmax (for perplexity), streamed
 * (does not retain the full logits matrix). is_int8 selects which weight
 * path (fp32 matvec vs int8 matvec) computes the matmuls; everything else
 * (norm/rope/softmax) is identical fp32 in both cases so the two are
 * directly comparable. Returns sum of -log(p(next_token)) in `nll_sum_out`
 * and count of top-1 matches in `top1_matches_out`; caller divides by
 * (ntok - 1) for mean NLL / agreement fraction.
 */
/* pred_ids_out, if non-NULL, must point to a caller-allocated buffer of
 * size (ntok - 1) uint32_t; filled with argmax(logits at pos) for
 * pos in [0, ntok-2], so two calls (use_int8=0 and use_int8=1) can be
 * diffed directly for int8-vs-fp32 top-1 agreement (independent of
 * ground truth). */
void cis2_eval_teacher_forced(const cis2_model *m, const cis2_qmodel *qm, int use_int8,
                               const uint32_t *tokens, size_t ntok,
                               double *nll_sum_out, size_t *top1_matches_out,
                               uint32_t *pred_ids_out);

/* spec-1: one greedy draft step. Recomputes the full forward pass over
 * tokens[0..ntok-1] (no KV cache -- this is the existing int8 path's
 * existing behavior, unchanged) and returns argmax(logits at the last
 * position), i.e. the int8 model's own greedy prediction of the next
 * token. Used by the speculative-decoding driver (model_spec.c) to
 * propose k draft tokens one at a time. Kept for reference/comparison;
 * cis2_qprocess_position (below) is what the driver actually uses, since
 * this O(ntok) per-call recompute makes whole-run drafting O(n_gen^2).
 */
uint32_t cis2_qdraft_next(const cis2_qmodel *qm, const uint32_t *tokens, size_t ntok);

/* spec-1: incremental (KV-cache) single-position int8 forward step,
 * mirroring model.c's cis2_process_position but with every weight matvec
 * (q/k/v/o/gate/up/down/lm_head) going through the int8/int32 path
 * (cis2_matvec_int8) instead of fp32 -- same KV-cache reuse discipline as
 * fast-1 (each (layer,position) computed exactly once), applied to the
 * draft model so proposing k tokens ahead costs O(k), not O(k*ntok).
 * `cache` must be a cis2_decode_kv_cache allocated/owned by the caller
 * (model.h's cis2_kv_cache_alloc/free -- the cache struct is just float
 * K/V arrays, independent of weight dtype, so the same type is reused for
 * both the fp32 target's cache and this int8 draft cache, as separate
 * instances). Returns malloc'd logits[vocab] if need_logits, else NULL. */
float *cis2_qprocess_position(const cis2_qmodel *qm, cis2_decode_kv_cache *cache,
                               size_t pos, uint32_t token_id, int need_logits);

#endif
