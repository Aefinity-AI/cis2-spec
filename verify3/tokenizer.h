/* tokenizer.h — byte-level BPE tokenizer, loaded from a HuggingFace
 * `tokenizers`-format tokenizer.json (spec §3.1). Written from scratch
 * using public knowledge of the tokenizers library's on-disk BPE schema
 * and the standard GPT-2 byte<->unicode mapping / BPE merge algorithm
 * (see verify3/CLEANROOM_LOG.md).
 */
#ifndef CIS2_V3_TOKENIZER_H
#define CIS2_V3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct cis2_tokenizer cis2_tokenizer;

/* Load and build the tokenizer from a tokenizer.json file's raw text. */
cis2_tokenizer *cis2_tokenizer_load(const char *json_text);
void cis2_tokenizer_free(cis2_tokenizer *t);

/* Encode `text` (raw UTF-8, no special tokens added — spec §3.3,
 * add_special_tokens=false) into token ids. Caller frees *out_ids. */
void cis2_tokenizer_encode(const cis2_tokenizer *t, const char *text,
                            uint32_t **out_ids, size_t *out_count);

#endif
