/* model.h — CIS-2 v0.2 model load + forward pass + witness/argmax digests. */
#ifndef CIS2_V3_MODEL_H
#define CIS2_V3_MODEL_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;
    int tie_word_embeddings;
    size_t vocab_size;
} cis2_config;

typedef struct {
    float *input_layernorm;      /* [hidden] */
    float *post_attention_layernorm; /* [hidden] */
    float *q_proj;   /* [hidden, hidden] */
    float *k_proj;   /* [n_kv*head_dim, hidden] */
    float *v_proj;   /* [n_kv*head_dim, hidden] */
    float *o_proj;   /* [hidden, hidden] */
    float *gate_proj; /* [inter, hidden] */
    float *up_proj;   /* [inter, hidden] */
    float *down_proj; /* [hidden, inter] */
    float *q_bias; /* optional, NULL if absent */
    float *k_bias;
    float *v_bias;
} cis2_layer;

typedef struct {
    cis2_config cfg;
    float *embed_tokens; /* [vocab, hidden] */
    cis2_layer *layers;
    float *norm_weight; /* [hidden] */
    float *lm_head; /* optional separate tensor; NULL if tied */
    float *inv_freq; /* [head_dim/2] */
    size_t head_dim;
    size_t half;
    size_t group;
} cis2_model;

int cis2_sha256_file(const char *path, uint8_t out[32]);

/* spec-1: incremental KV-cache decode state (model.c) exposed so the
 * speculative-decoding driver (model_spec.c) can drive the fp32 target
 * path position-by-position and in verify-batches, without duplicating
 * or altering the forward-pass math. See model.c "---- incremental
 * (KV-cache) decode state ----" for the invariant this relies on: each
 * (layer, position)'s output depends only on its own residual stream and
 * already-finalized K/V for positions <= it. */
typedef struct {
    float **cache_k; /* [num_layers][cap * nkv_dim] */
    float **cache_v; /* [num_layers][cap * nkv_dim] */
} cis2_decode_kv_cache;

cis2_decode_kv_cache *cis2_kv_cache_alloc(const cis2_model *m, size_t cap);
void cis2_kv_cache_free(cis2_decode_kv_cache *c, const cis2_model *m);

/* Processes exactly one new sequence position (same contract as the
 * internal process_position() this wraps). Returns malloc'd logits[vocab]
 * if need_logits, else NULL. */
float *cis2_process_position(const cis2_model *m, cis2_decode_kv_cache *cache,
                              size_t pos, uint32_t token_id, int need_logits);

/* spec-1: batched target verification. Processes `batch_n` NEW,
 * consecutive positions [base_pos, base_pos+batch_n) in one pass,
 * appending each position's K/V into `cache` (same as calling
 * cis2_process_position for each position in increasing order, with
 * cache reads/writes in the same order) but batches every weight matvec
 * (q/k/v/o/gate/up/down/lm_head) across the batch_n positions so each
 * weight matrix is streamed from memory once per layer instead of
 * batch_n times (see mathpin.c cis2_matvec_batch). Per-position math
 * (rmsnorm/rope/attention-softmax) is unchanged and independent per
 * position, so batching the matvec loop order does not change any
 * floating-point value versus calling cis2_process_position batch_n
 * times in order -- only which memory-access order computes them.
 * token_ids[i] is the token fed at position base_pos+i (i.e. what
 * cis2_process_position would receive as token_id at that position).
 * out_logits[i] (caller-allocated, vocab floats each) receives that
 * position's logits (always computed -- this function always needs
 * logits at every position in the batch for the caller's accept/reject
 * comparisons). */
void cis2_process_positions_batch(const cis2_model *m, cis2_decode_kv_cache *cache,
                                   size_t base_pos, const uint32_t *token_ids,
                                   size_t batch_n, float **out_logits);

uint32_t cis2_argmax(const float *logits, size_t n);

/* E15k debug-only: enable/disable the CIS2_DUMP per-layer state
 * side-channel (see model.c dump_line()). Not part of any normative
 * digest; gated on env var CIS2_DUMP_LAYERS in main.c. */
void cis2_set_dump_layers(int enabled);

/* config_json_text and safetensors_path: config is parsed from text,
 * weights loaded (and bf16->f32 widened) from the safetensors file. */
cis2_model *cis2_model_load(const char *safetensors_path, const char *config_json_text);
void cis2_model_free(cis2_model *m);

typedef struct {
    uint32_t *generated_ids; /* [n_gen] */
    size_t n_gen;
    uint8_t witness_digest[32];
    uint8_t argmax_digest[32];
    uint8_t table_digest[32];
    uint8_t inv_freq_table_digest[32];
} cis2_run_result;

/* Runs the full greedy decode (spec §3.4/§11/§12) once. weights_sha256,
 * tokenizer_sha256, config_sha256 are the raw 32-byte artifact hashes fed
 * into the witness chain (§12.1 items 3-5). */
void cis2_run_decode(const cis2_model *m,
                       const uint32_t *prompt_ids, size_t n_prompt, size_t n_gen,
                       const uint8_t weights_sha256[32],
                       const uint8_t tokenizer_sha256[32],
                       const uint8_t config_sha256[32],
                       cis2_run_result *out);

#endif
