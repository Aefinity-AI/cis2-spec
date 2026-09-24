/* model_ternary.h — ternary (BitNet-b1.58-style) per-output-channel-weight /
 * int32-accumulation forward pass built on top of the fp32 cis2_model
 * (QUEUE item small-2: TERNARY vs INT8 AT MATCHED BYTES). Only the
 * weight-matmul reductions (q/k/v/o/gate/up/down/embed-as-lm_head) are
 * ternary-quantized with int32 accumulation; RMSNorm, RoPE, softmax and the
 * attention score/value reductions are untouched fp32 (same pinned
 * primitives as verify3/mathpin.c) — this file adds a THIRD path; it does
 * not modify model.c, mathpin.c, model_int8.c, quant.c, main.c, main_int8.c
 * or the existing fp32/int8 digests.
 */
#ifndef CIS2_V3_MODEL_TERNARY_H
#define CIS2_V3_MODEL_TERNARY_H

#include <stdint.h>
#include <stddef.h>
#include "model.h"
#include "quant_ternary.h"

typedef struct {
    cis2_tqmat q_proj, k_proj, v_proj, o_proj;
    cis2_tqmat gate_proj, up_proj, down_proj;
} cis2_tqlayer;

typedef struct {
    const cis2_model *base; /* not owned */
    cis2_tqmat embed; /* [vocab, hidden]; also used as lm_head when tied */
    cis2_tqmat lm_head; /* only populated if !tie_word_embeddings */
    int has_separate_lm_head;
    cis2_tqlayer *layers;
    size_t n_bytes_quantized; /* total bytes of packed[] + scale[] arrays, for size reporting */
} cis2_tqmodel;

/* Builds the ternary-quantized weight cache from an already-loaded fp32 model. */
cis2_tqmodel *cis2_tqmodel_build(const cis2_model *m);
void cis2_tqmodel_free(cis2_tqmodel *qm);

typedef struct {
    uint32_t *generated_ids;
    size_t n_gen;
    uint8_t witness_digest[32];
    uint8_t argmax_digest[32];
} cis2_tqrun_result;

/* Same witness-chain construction as cis2_run_decode (model.c) / small-1's
 * cis2_qrun_decode, but every weight matmul in the forward pass runs
 * through cis2_matvec_ternary instead of cis2_matvec / cis2_matvec_int8. */
void cis2_tqrun_decode(const cis2_tqmodel *qm,
                        const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen,
                        const uint8_t weights_sha256[32],
                        const uint8_t tokenizer_sha256[32],
                        const uint8_t config_sha256[32],
                        cis2_tqrun_result *out);

/* Teacher-forced single pass, mirrors cis2_eval_teacher_forced (model_int8.c)
 * but dispatches to the ternary matvec when use_ternary != 0, fp32 otherwise
 * (independent fp32 baseline path, same forward skeleton, directly
 * comparable to small-1's fp32-vs-int8 numbers). */
void cis2_eval_teacher_forced_ternary(const cis2_model *m, const cis2_tqmodel *qm, int use_ternary,
                                       const uint32_t *tokens, size_t ntok,
                                       double *nll_sum_out, size_t *top1_matches_out,
                                       uint32_t *pred_ids_out);

#endif
