/* malformed_driver.c — regression driver for the safetensors loader.
 *
 * Each path given on the command line is fed to cis2_model_load() in a child
 * process. A malformed file must be rejected through the loader's fatal path
 * (exit status 95): any signal, sanitizer report (non-95 status) or normal
 * return from the loader counts as a failure. Inputs are treated as data only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../model.h"

#if defined(__SANITIZE_ADDRESS__)
#define CIS2_HAVE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CIS2_HAVE_ASAN 1
#endif
#endif
#ifdef CIS2_HAVE_ASAN
/* The loader exits without freeing on a fatal path; do not report that as a leak. */
const char *__asan_default_options(void) { return "detect_leaks=0:exitcode=86:abort_on_error=0"; }
#endif

static const char *CONFIG =
    "{\"hidden_size\":576,\"intermediate_size\":1536,\"num_hidden_layers\":30,"
    "\"num_attention_heads\":9,\"num_key_value_heads\":3,\"rms_norm_eps\":1e-05,"
    "\"rope_theta\":100000,\"tie_word_embeddings\":true,\"vocab_size\":49152}";

int main(int argc, char **argv)
{
    int bad = 0;
    for (int i = 1; i < argc; i++) {
        fflush(NULL);
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 2; }
        if (pid == 0) {
            if (!getenv("MALFORMED_VERBOSE")) /* loader diagnostics are not the test output */
                freopen("/dev/null", "w", stderr);
            cis2_model *m = cis2_model_load(argv[i], CONFIG);
            _exit(m ? 0 : 1); /* returned: the file was accepted */
        }
        int st = 0;
        waitpid(pid, &st, 0);
        if (WIFEXITED(st) && WEXITSTATUS(st) == 95) {
            printf("ok:   %s rejected\n", argv[i]);
        } else {
            if (WIFSIGNALED(st)) printf("FAIL: %s killed by signal %d\n", argv[i], WTERMSIG(st));
            else printf("FAIL: %s exit status %d (95 expected)\n", argv[i], WEXITSTATUS(st));
            bad++;
        }
    }
    printf("%s: %d file(s), %d failure(s)\n", bad ? "FAIL" : "PASS", argc - 1, bad);
    return bad ? 1 : 0;
}
