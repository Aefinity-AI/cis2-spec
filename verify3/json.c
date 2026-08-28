/* json.c — minimal hand-written recursive-descent JSON reader.
 * Written from scratch for this task (not vendored).
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "json.h"

typedef struct {
    const char *p;
} parser_state;

static void skip_ws(parser_state *s)
{
    while (*s->p == ' ' || *s->p == '\t' || *s->p == '\n' || *s->p == '\r') s->p++;
}

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

static unsigned int hex4(const char *p)
{
    unsigned int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
    }
    return v;
}

static char *parse_string_raw(parser_state *s, size_t *out_len)
{
    if (*s->p != '"') return NULL;
    s->p++;
    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
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
                    s->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && s->p[0] == '\\' && s->p[1] == 'u') {
                        unsigned int lo = hex4(s->p + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            s->p += 6;
                        }
                    }
                    utf8_append(&buf, &len, &cap, cp);
                    break;
                }
                default: s->p++; break;
            }
        } else {
            if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
            buf[len++] = (char)c;
            s->p++;
        }
    }
    if (*s->p == '"') s->p++;
    if (len + 1 > cap) { cap = len + 1; buf = realloc(buf, cap); }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

static json_value *parse_object(parser_state *s)
{
    json_value *v = calloc(1, sizeof(json_value));
    v->type = JSON_OBJECT;
    s->p++; /* { */
    skip_ws(s);
    size_t cap = 8, count = 0;
    char **keys = malloc(cap * sizeof(char*));
    json_value **vals = malloc(cap * sizeof(json_value*));
    if (*s->p == '}') { s->p++; v->u.object.keys = keys; v->u.object.vals = vals; v->u.object.count = 0; return v; }
    while (1) {
        skip_ws(s);
        size_t klen;
        char *key = parse_string_raw(s, &klen);
        skip_ws(s);
        if (*s->p == ':') s->p++;
        skip_ws(s);
        json_value *val = parse_value(s);
        if (count == cap) {
            cap *= 2;
            keys = realloc(keys, cap * sizeof(char*));
            vals = realloc(vals, cap * sizeof(json_value*));
        }
        keys[count] = key;
        vals[count] = val;
        count++;
        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        break;
    }
    skip_ws(s);
    if (*s->p == '}') s->p++;
    v->u.object.keys = keys;
    v->u.object.vals = vals;
    v->u.object.count = count;
    return v;
}

static json_value *parse_array(parser_state *s)
{
    json_value *v = calloc(1, sizeof(json_value));
    v->type = JSON_ARRAY;
    s->p++; /* [ */
    skip_ws(s);
    size_t cap = 8, count = 0;
    json_value **items = malloc(cap * sizeof(json_value*));
    if (*s->p == ']') { s->p++; v->u.array.items = items; v->u.array.count = 0; return v; }
    while (1) {
        skip_ws(s);
        json_value *val = parse_value(s);
        if (count == cap) { cap *= 2; items = realloc(items, cap * sizeof(json_value*)); }
        items[count++] = val;
        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        break;
    }
    skip_ws(s);
    if (*s->p == ']') s->p++;
    v->u.array.items = items;
    v->u.array.count = count;
    return v;
}

static json_value *parse_value(parser_state *s)
{
    skip_ws(s);
    if (*s->p == '"') {
        size_t len;
        char *str = parse_string_raw(s, &len);
        json_value *v = calloc(1, sizeof(json_value));
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
        v->type = JSON_BOOL; v->u.boolean = 1; return v;
    } else if (strncmp(s->p, "false", 5) == 0) {
        s->p += 5;
        json_value *v = calloc(1, sizeof(json_value));
        v->type = JSON_BOOL; v->u.boolean = 0; return v;
    } else if (strncmp(s->p, "null", 4) == 0) {
        s->p += 4;
        json_value *v = calloc(1, sizeof(json_value));
        v->type = JSON_NULL; return v;
    } else {
        /* number */
        const char *start = s->p;
        if (*s->p == '-' || *s->p == '+') s->p++;
        while (isdigit((unsigned char)*s->p) || *s->p == '.' || *s->p == 'e' || *s->p == 'E' ||
               *s->p == '+' || *s->p == '-') s->p++;
        char numbuf[64];
        size_t n = (size_t)(s->p - start);
        if (n >= sizeof(numbuf)) n = sizeof(numbuf) - 1;
        memcpy(numbuf, start, n);
        numbuf[n] = '\0';
        json_value *v = calloc(1, sizeof(json_value));
        v->type = JSON_NUMBER;
        v->u.number = atof(numbuf);
        return v;
    }
}

json_value *json_parse(const char *text)
{
    parser_state s;
    s.p = text;
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
        if (strcmp(v->u.object.keys[i], key) == 0) return v->u.object.vals[i];
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
