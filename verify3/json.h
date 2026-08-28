/* json.h — minimal hand-written recursive-descent JSON reader.
 * Written from scratch for this task (not vendored). Sufficient to parse
 * safetensors JSON headers and tokenizer.json.
 */
#ifndef CIS2_V3_JSON_H
#define CIS2_V3_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT
} json_type;

typedef struct json_value {
    json_type type;
    union {
        int boolean;
        double number;
        struct { char *ptr; size_t len; } string;
        struct { struct json_value **items; size_t count; } array;
        struct { char **keys; struct json_value **vals; size_t count; } object;
    } u;
} json_value;

/* Parse a NUL-terminated JSON text. Returns NULL on parse error. */
json_value *json_parse(const char *text);
void json_free(json_value *v);

/* Object lookup helper: returns NULL if not found or v is not an object. */
json_value *json_obj_get(const json_value *v, const char *key);

/* Convenience accessors (return 0/NULL-safe defaults if wrong type). */
const char *json_as_cstring(const json_value *v); /* NUL-terminated copy owned by value */
double json_as_number(const json_value *v);
size_t json_array_len(const json_value *v);
json_value *json_array_get(const json_value *v, size_t idx);

#endif
