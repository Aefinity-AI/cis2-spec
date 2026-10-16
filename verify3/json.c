/* json.c — minimal hand-written recursive-descent JSON reader.
 * Written from scratch for this task (not vendored).
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "json.h"

typedef struct {
    const char *p;
    int depth;
} parser_state;

static void skip_ws(parser_state *s)
{
    while (*s->p == ' ' || *s->p == '\t' || *s->p == '\n' || *s->p == '\r') s->p++;
}

#define JSON_MAX_DEPTH 256

static json_value *parse_value(parser_state *s);

static void utf8_append(char **buf, size_t *len, size_t *cap, unsigned int cp)
{
    char tmp[4];
    int n = 0;
    if (cp <= 0x7F) {
        tmp[0] = (char)cp; n = 1;
    } else if (cp <= 0x7FF) {
        tmp[0] = (char)(0xC0 | (cp >> 6));
        tmp[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp <= 0xFFFF) {
        tmp[0] = (char)(0xE0 | (cp >> 12));
        tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = (char)(0xF0 | (cp >> 18));
        tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    if (*len + (size_t)n + 1 > *cap) {
        *cap = (*cap + n + 1) * 2;
        *buf = realloc(*buf, *cap);
    }
    memcpy(*buf + *len, tmp, (size_t)n);
    *len += (size_t)n;
}

/* Returns 0xFFFFFFFF if fewer than 4 hex digits follow (never reads past NUL). */
static unsigned int hex4(const char *p)
{
    unsigned int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0xFFFFFFFFu;
    }
    return v;
}

static char *parse_string_raw(parser_state *s, size_t *out_len)
{
    if (*s->p != '"') return NULL;
    s->p++;
    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    while (*s->p && *s->p != '"') {
        unsigned char c = (unsigned char)*s->p;
        if (c == '\\') {
            s->p++;
            char esc = *s->p;
            switch (esc) {
                case '"': utf8_append(&buf, &len, &cap, '"'); s->p++; break;
                case '\\': utf8_append(&buf, &len, &cap, '\\'); s->p++; break;
                case '/': utf8_append(&buf, &len, &cap, '/'); s->p++; break;
                case 'b': utf8_append(&buf, &len, &cap, '\b'); s->p++; break;
                case 'f': utf8_append(&buf, &len, &cap, '\f'); s->p++; break;
                case 'n': utf8_append(&buf, &len, &cap, '\n'); s->p++; break;
                case 'r': utf8_append(&buf, &len, &cap, '\r'); s->p++; break;
                case 't': utf8_append(&buf, &len, &cap, '\t'); s->p++; break;
                case 'u': {
                    s->p++;
                    unsigned int cp = hex4(s->p);
                    if (cp == 0xFFFFFFFFu) { free(buf); return NULL; }
                    s->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && s->p[0] == '\\' && s->p[1] == 'u') {
                        unsigned int lo = hex4(s->p + 2);
                        if (lo != 0xFFFFFFFFu && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            s->p += 6;
                        }
                    }
                    utf8_append(&buf, &len, &cap, cp);
                    break;
                }
                default: free(buf); return NULL; /* invalid escape or NUL */
            }
        } else {
            if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
            buf[len++] = (char)c;
            s->p++;
        }
    }
    if (*s->p != '"') { free(buf); return NULL; } /* unterminated string */
    s->p++;
    if (len + 1 > cap) { cap = len + 1; buf = realloc(buf, cap); }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

static json_value *parse_object(parser_state *s)
{
    json_value *v = calloc(1, sizeof(json_value));
    if (!v) return NULL;
    v->type = JSON_OBJECT;
    s->p++; /* { */
    skip_ws(s);
    size_t cap = 8, count = 0;
    char **keys = malloc(cap * sizeof(char*));
    json_value **vals = malloc(cap * sizeof(json_value*));
    if (!keys || !vals) { free(keys); free(vals); free(v); return NULL; }
    v->u.object.keys = keys; v->u.object.vals = vals; v->u.object.count = 0;
    if (*s->p == '}') { s->p++; return v; }
    while (1) {
        skip_ws(s);
        size_t klen;
        char *key = parse_string_raw(s, &klen);
        if (!key) goto fail; /* keys must be strings: never store a NULL key */
        skip_ws(s);
        if (*s->p != ':') { free(key); goto fail; }
        s->p++;
        skip_ws(s);
        json_value *val = parse_value(s);
        if (!val) { free(key); goto fail; }
        if (count == cap) {
            size_t ncap = cap * 2;
            char **nk = realloc(keys, ncap * sizeof(char*));
            if (nk) { keys = nk; v->u.object.keys = nk; }
            json_value **nv = realloc(vals, ncap * sizeof(json_value*));
            if (nv) { vals = nv; v->u.object.vals = nv; }
            if (!nk || !nv) { free(key); json_free(val); goto fail; }
            cap = ncap;
        }
        keys[count] = key;
        vals[count] = val;
        count++;
        v->u.object.count = count;
        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        break;
    }
    skip_ws(s);
    if (*s->p != '}') goto fail;
    s->p++;
    return v;
fail:
    json_free(v);
    return NULL;
}

static json_value *parse_array(parser_state *s)
{
    json_value *v = calloc(1, sizeof(json_value));
    if (!v) return NULL;
    v->type = JSON_ARRAY;
    s->p++; /* [ */
    skip_ws(s);
    size_t cap = 8, count = 0;
    json_value **items = malloc(cap * sizeof(json_value*));
    if (!items) { free(v); return NULL; }
    v->u.array.items = items; v->u.array.count = 0;
    if (*s->p == ']') { s->p++; return v; }
    while (1) {
        skip_ws(s);
        json_value *val = parse_value(s);
        if (!val) goto fail;
        if (count == cap) {
            json_value **ni = realloc(items, cap * 2 * sizeof(json_value*));
            if (!ni) { json_free(val); goto fail; }
            items = ni; v->u.array.items = ni; cap *= 2;
        }
        items[count++] = val;
        v->u.array.count = count;
        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        break;
    }
    skip_ws(s);
    if (*s->p != ']') goto fail;
    s->p++;
    return v;
fail:
    json_free(v);
    return NULL;
}

static json_value *parse_value_inner(parser_state *s)
{
    skip_ws(s);
    if (*s->p == '"') {
        size_t len;
        char *str = parse_string_raw(s, &len);
        if (!str) return NULL;
        json_value *v = calloc(1, sizeof(json_value));
        if (!v) { free(str); return NULL; }
        v->type = JSON_STRING;
        v->u.string.ptr = str;
        v->u.string.len = len;
        return v;
    } else if (*s->p == '{') {
        return parse_object(s);
    } else if (*s->p == '[') {
        return parse_array(s);
    } else if (strncmp(s->p, "true", 4) == 0) {
        s->p += 4;
        json_value *v = calloc(1, sizeof(json_value));
        if (!v) return NULL;
        v->type = JSON_BOOL; v->u.boolean = 1; return v;
    } else if (strncmp(s->p, "false", 5) == 0) {
        s->p += 5;
        json_value *v = calloc(1, sizeof(json_value));
        if (!v) return NULL;
        v->type = JSON_BOOL; v->u.boolean = 0; return v;
    } else if (strncmp(s->p, "null", 4) == 0) {
        s->p += 4;
        json_value *v = calloc(1, sizeof(json_value));
        if (!v) return NULL;
        v->type = JSON_NULL; return v;
    } else {
        /* number */
        const char *start = s->p;
        if (*s->p == '-' || *s->p == '+') s->p++;
        while (isdigit((unsigned char)*s->p) || *s->p == '.' || *s->p == 'e' || *s->p == 'E' ||
               *s->p == '+' || *s->p == '-') s->p++;
        char numbuf[64];
        size_t n = (size_t)(s->p - start);
        if (n == 0) return NULL; /* not a value */
        if (n >= sizeof(numbuf)) n = sizeof(numbuf) - 1;
        memcpy(numbuf, start, n);
        numbuf[n] = '\0';
        json_value *v = calloc(1, sizeof(json_value));
        if (!v) return NULL;
        v->type = JSON_NUMBER;
        v->u.number = atof(numbuf);
        return v;
    }
}

static json_value *parse_value(parser_state *s)
{
    if (s->depth >= JSON_MAX_DEPTH) return NULL; /* bound recursion */
    s->depth++;
    json_value *v = parse_value_inner(s);
    s->depth--;
    return v;
}

json_value *json_parse(const char *text)
{
    parser_state s;
    s.p = text;
    s.depth = 0;
    if (!text) return NULL;
    skip_ws(&s);
    if (!*s.p) return NULL;
    return parse_value(&s);
}

void json_free(json_value *v)
{
    if (!v) return;
    switch (v->type) {
        case JSON_STRING: free(v->u.string.ptr); break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->u.array.count; i++) json_free(v->u.array.items[i]);
            free(v->u.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->u.object.count; i++) {
                free(v->u.object.keys[i]);
                json_free(v->u.object.vals[i]);
            }
            free(v->u.object.keys);
            free(v->u.object.vals);
            break;
        default: break;
    }
    free(v);
}

json_value *json_obj_get(const json_value *v, const char *key)
{
    if (!v || v->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < v->u.object.count; i++) {
        if (v->u.object.keys[i] && key && strcmp(v->u.object.keys[i], key) == 0) return v->u.object.vals[i];
    }
    return NULL;
}

const char *json_as_cstring(const json_value *v)
{
    if (!v || v->type != JSON_STRING) return NULL;
    return v->u.string.ptr;
}

double json_as_number(const json_value *v)
{
    if (!v || v->type != JSON_NUMBER) return 0.0;
    return v->u.number;
}

size_t json_array_len(const json_value *v)
{
    if (!v || v->type != JSON_ARRAY) return 0;
    return v->u.array.count;
}

json_value *json_array_get(const json_value *v, size_t idx)
{
    if (!v || v->type != JSON_ARRAY || idx >= v->u.array.count) return NULL;
    return v->u.array.items[idx];
}
