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

/* E15k debug-only: enable/disable the CIS2_DUMP per-layer state
 * side-channel (see model.c dump_line()). Not part of any normative
 * digest; gated on env var CIS2_DUMP_LAYERS in main.c. */
void cis2_set_dump_layers(int enabled);

/* config_json_text and safetensors_path: config is parsed from text,
 * weights loaded (and bf16->f32 widened) from the safetensors file. */
cis2_model *cis2_model_load(const char *safetensors_path, const char *config_json_text);
void cis2_model_free(cis2_model *m);

/* xb-1: logit-lens top-5 entry (token id + fp32 logit value), computed by
 * projecting a layer's post-block hidden state through the model's real
 * final-norm + unembedding head (same math the model already uses to
 * produce the final logits). Not a normative digest by itself; see
 * cis2_run_result.layer_lens_digest below. */
typedef struct {
    uint32_t token_id;
    float logit;
} cis2_lens_entry;

#define CIS2_LENS_TOPK 5

typedef struct {
    uint32_t *generated_ids; /* [n_gen] */
    size_t n_gen;
    uint8_t witness_digest[32];
    uint8_t argmax_digest[32];
    uint8_t table_digest[32];
    uint8_t inv_freq_table_digest[32];

    /* xb-1 (additive, non-normative): per-layer logit-lens top-5 table for
     * the fixed prompt (computed at the last prompt position, step 0 of
     * decode, using the model's real unembedding weights in fp32).
     * layer_lens_table is [num_hidden_layers * CIS2_LENS_TOPK] entries,
     * row-major by layer then rank (0=highest logit). Caller frees.
     * layer_lens_digest is a SEPARATE sha256 over the serialized table
     * bytes -- it is deliberately NOT folded into witness_digest's byte
     * stream, so witness/argmax/table/inv_freq_table digests remain
     * bit-identical to pre-xb-1 golden values. This is the "additive
     * sibling digest" architecture: the receipt gains a new, independently
     * verifiable field rather than perturbing the existing witness chain. */
    uint8_t layer_lens_digest[32];
    cis2_lens_entry *layer_lens_table;
    size_t layer_lens_n_layers;
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
