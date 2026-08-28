/* tokenizer.c — byte-level BPE tokenizer. See tokenizer.h header comment
 * and verify3/CLEANROOM_LOG.md for provenance notes.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tokenizer.h"
#include "json.h"

/* ---- tiny string->long hashmap (open addressing, FNV-1a) ---- */

typedef struct { char *key; long value; int used; } hm_entry;
typedef struct { hm_entry *entries; size_t cap; size_t count; } hashmap;

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static void hm_init(hashmap *m, size_t initial_cap)
{
    m->cap = initial_cap;
    m->count = 0;
    m->entries = calloc(m->cap, sizeof(hm_entry));
}

static void hm_grow(hashmap *m);

static void hm_put(hashmap *m, const char *key, long value)
{
    if ((m->count + 1) * 2 > m->cap) hm_grow(m);
    uint64_t h = fnv1a(key);
    size_t idx = (size_t)(h % m->cap);
    while (m->entries[idx].used) {
        if (strcmp(m->entries[idx].key, key) == 0) {
            m->entries[idx].value = value;
            return;
        }
        idx = (idx + 1) % m->cap;
    }
    m->entries[idx].key = strdup(key);
    m->entries[idx].value = value;
    m->entries[idx].used = 1;
    m->count++;
}

static void hm_grow(hashmap *m)
{
    hashmap nm;
    hm_init(&nm, m->cap * 2 + 16);
    for (size_t i = 0; i < m->cap; i++) {
        if (m->entries[i].used) hm_put(&nm, m->entries[i].key, m->entries[i].value);
    }
    free(m->entries);
    *m = nm;
}

static int hm_get(const hashmap *m, const char *key, long *out_value)
{
    if (m->cap == 0) return 0;
    uint64_t h = fnv1a(key);
    size_t idx = (size_t)(h % m->cap);
    size_t start = idx;
    while (m->entries[idx].used) {
        if (strcmp(m->entries[idx].key, key) == 0) {
            *out_value = m->entries[idx].value;
            return 1;
        }
        idx = (idx + 1) % m->cap;
        if (idx == start) break;
    }
    return 0;
}

static void hm_free(hashmap *m)
{
    for (size_t i = 0; i < m->cap; i++) {
        if (m->entries[i].used) free(m->entries[i].key);
    }
    free(m->entries);
}

/* ---- GPT-2 byte<->unicode mapping ---- */

struct cis2_tokenizer {
    hashmap vocab;   /* byte-level-encoded token string -> id */
    hashmap merges;  /* "left\x01right" -> rank */
    uint32_t byte_encoder_cp[256]; /* byte -> unicode codepoint */
};

static void build_byte_encoder(uint32_t cp[256])
{
    int is_bs[256] = {0};
    /* bs = range(ord('!'),ord('~')+1) + range(ord(0xA1),ord(0xAC)+1) + range(ord(0xAE),ord(0xFF)+1) */
    for (int b = '!'; b <= '~'; b++) is_bs[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++) is_bs[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) is_bs[b] = 1;

    for (int b = 0; b < 256; b++) {
        if (is_bs[b]) cp[b] = (uint32_t)b;
        else cp[b] = 0; /* filled below */
    }
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        if (!is_bs[b]) {
            cp[b] = 256u + n;
            n++;
        }
    }
}

static size_t utf8_encode_cp(uint32_t cp, char *out)
{
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    } else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/* ---- load ---- */

cis2_tokenizer *cis2_tokenizer_load(const char *json_text)
{
    json_value *root = json_parse(json_text);
    if (!root) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: failed to parse tokenizer.json\n");
        exit(96);
    }
    json_value *model = json_obj_get(root, "model");
    json_value *vocab_j = json_obj_get(model, "vocab");
    json_value *merges_j = json_obj_get(model, "merges");
    if (!vocab_j || !merges_j) {
        fprintf(stderr, "CIS2_VERIFY3 FATAL: tokenizer.json missing model.vocab/model.merges\n");
        exit(96);
    }

    cis2_tokenizer *t = calloc(1, sizeof(cis2_tokenizer));
    build_byte_encoder(t->byte_encoder_cp);

    hm_init(&t->vocab, vocab_j->u.object.count * 2 + 16);
    for (size_t i = 0; i < vocab_j->u.object.count; i++) {
        hm_put(&t->vocab, vocab_j->u.object.keys[i], (long)json_as_number(vocab_j->u.object.vals[i]));
    }

    size_t nmerges = json_array_len(merges_j);
    hm_init(&t->merges, nmerges * 2 + 16);
    for (size_t i = 0; i < nmerges; i++) {
        json_value *m = json_array_get(merges_j, i);
        char keybuf[512];
        if (m->type == JSON_STRING) {
            /* "left right" format */
            const char *s = json_as_cstring(m);
            const char *sp = strchr(s, ' ');
            if (sp) {
                size_t llen = (size_t)(sp - s);
                if (llen > 250) llen = 250;
                char left[256];
                memcpy(left, s, llen);
                left[llen] = '\0';
                snprintf(keybuf, sizeof(keybuf), "%s\x01%s", left, sp + 1);
            } else {
                snprintf(keybuf, sizeof(keybuf), "%s\x01", s);
            }
        } else if (m->type == JSON_ARRAY && json_array_len(m) == 2) {
            const char *l = json_as_cstring(json_array_get(m, 0));
            const char *r = json_as_cstring(json_array_get(m, 1));
            snprintf(keybuf, sizeof(keybuf), "%s\x01%s", l, r);
        } else {
            continue;
        }
        hm_put(&t->merges, keybuf, (long)i);
    }

    json_free(root);
    return t;
}

void cis2_tokenizer_free(cis2_tokenizer *t)
{
    if (!t) return;
    hm_free(&t->vocab);
    hm_free(&t->merges);
    free(t);
}

/* ---- pre-tokenizer: ASCII-sufficient approximation of the GPT-2/
 * ByteLevel regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
 * ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`. See CLEANROOM_LOG.md / SPEC_GAPS. */

typedef struct { size_t start, end; } span;

static size_t pretokenize(const char *text, span *out, size_t max_out)
{
    static const char *contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    size_t n = strlen(text);
    size_t i = 0, count = 0;
    while (i < n && count < max_out) {
        size_t start = i;
        int matched_contraction = 0;
        for (int c = 0; c < 7; c++) {
            size_t clen = strlen(contractions[c]);
            if (i + clen <= n && strncmp(text + i, contractions[c], clen) == 0) {
                i += clen;
                matched_contraction = 1;
                break;
            }
        }
        if (matched_contraction) {
            out[count].start = start; out[count].end = i; count++;
            continue;
        }

        int has_space = 0;
        if ((unsigned char)text[i] == ' ') { has_space = 1; i++; }

        if (i < n && isalpha((unsigned char)text[i])) {
            while (i < n && isalpha((unsigned char)text[i])) i++;
            out[count].start = start; out[count].end = i; count++;
            continue;
        }
        if (i < n && isdigit((unsigned char)text[i])) {
            while (i < n && isdigit((unsigned char)text[i])) i++;
            out[count].start = start; out[count].end = i; count++;
            continue;
        }
        if (i < n && !isspace((unsigned char)text[i])) {
            while (i < n && !isspace((unsigned char)text[i]) &&
                   !isalpha((unsigned char)text[i]) && !isdigit((unsigned char)text[i])) i++;
            out[count].start = start; out[count].end = i; count++;
            continue;
        }
        /* whitespace-only run (covers the has_space-but-nothing-followed case too) */
        if (!has_space) {
            /* i currently sits on whitespace since none of the branches above matched */
        }
        while (i < n && isspace((unsigned char)text[i])) i++;
        if (i == start) { i++; } /* safety: never stall */
        out[count].start = start; out[count].end = i; count++;
    }
    return count;
}

/* ---- BPE merge on one pre-tokenized word ---- */

static void encode_word(const cis2_tokenizer *t, const char *word, size_t wlen,
                          uint32_t **ids, size_t *ids_count, size_t *ids_cap)
{
    /* symbols[i] = malloc'd NUL-terminated byte-level-encoded UTF-8 string */
    size_t nsym = wlen;
    char **symbols = malloc(nsym * sizeof(char*));
    for (size_t i = 0; i < wlen; i++) {
        unsigned char byte = (unsigned char)word[i];
        uint32_t cp = t->byte_encoder_cp[byte];
        char buf[5] = {0};
        size_t l = utf8_encode_cp(cp, buf);
        symbols[i] = malloc(l + 1);
        memcpy(symbols[i], buf, l);
        symbols[i][l] = '\0';
    }

    while (nsym > 1) {
        long best_rank = -1;
        size_t best_i = 0;
        int found = 0;
        for (size_t i = 0; i + 1 < nsym; i++) {
            char keybuf[512];
            snprintf(keybuf, sizeof(keybuf), "%s\x01%s", symbols[i], symbols[i + 1]);
            long rank;
            if (hm_get(&t->merges, keybuf, &rank)) {
                if (!found || rank < best_rank) {
                    best_rank = rank;
                    best_i = i;
                    found = 1;
                }
            }
        }
        if (!found) break;
        /* merge symbols[best_i] and symbols[best_i+1] */
        size_t l1 = strlen(symbols[best_i]);
        size_t l2 = strlen(symbols[best_i + 1]);
        char *merged = malloc(l1 + l2 + 1);
        memcpy(merged, symbols[best_i], l1);
        memcpy(merged + l1, symbols[best_i + 1], l2);
        merged[l1 + l2] = '\0';
        free(symbols[best_i]);
        free(symbols[best_i + 1]);
        symbols[best_i] = merged;
        for (size_t j = best_i + 1; j + 1 < nsym; j++) symbols[j] = symbols[j + 1];
        nsym--;
    }

    for (size_t i = 0; i < nsym; i++) {
        long id;
        if (!hm_get(&t->vocab, symbols[i], &id)) {
            fprintf(stderr, "CIS2_VERIFY3 FATAL: token '%s' not found in vocab\n", symbols[i]);
            exit(96);
        }
        if (*ids_count == *ids_cap) {
            *ids_cap = (*ids_cap) * 2 + 8;
            *ids = realloc(*ids, (*ids_cap) * sizeof(uint32_t));
        }
        (*ids)[(*ids_count)++] = (uint32_t)id;
        free(symbols[i]);
    }
    free(symbols);
}

void cis2_tokenizer_encode(const cis2_tokenizer *t, const char *text,
                            uint32_t **out_ids, size_t *out_count)
{
    span spans[4096];
    size_t nspans = pretokenize(text, spans, 4096);
    uint32_t *ids = NULL;
    size_t count = 0, cap = 0;
    for (size_t s = 0; s < nspans; s++) {
        size_t len = spans[s].end - spans[s].start;
        if (len == 0) continue;
        encode_word(t, text + spans[s].start, len, &ids, &count, &cap);
    }
    *out_ids = ids;
    *out_count = count;
}
