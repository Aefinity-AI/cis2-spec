# CIS-2 v0.2 clean-room verify3 (C11).
# Spec §1.4/1.5: no FMA, no fast-math, no reassociation.
CC ?= gcc
UNAME_M := $(shell uname -m)

CFLAGS_COMMON = -O2 -ffp-contract=off -fno-fast-math -fexcess-precision=standard -std=c11 -Wall -Wextra

ifeq ($(UNAME_M),x86_64)
CFLAGS_ISA = -mno-fma
else ifeq ($(UNAME_M),aarch64)
CFLAGS_ISA = -march=armv8-a
else
CFLAGS_ISA =
endif

CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_ISA)
LDFLAGS = -lm

SRCS = main.c model.c mathpin.c tokenizer.c json.c sha256.c
OBJS = $(SRCS:.c=.o)
BIN = cis2_verify3

SELFTEST_SRCS = selftest.c model.c mathpin.c tokenizer.c json.c sha256.c
SELFTEST_OBJS_ALL = $(SELFTEST_SRCS:.c=.o)

# small-1: int8 per-channel weight quantization / int32-accumulation path.
INT8_SRCS = main_int8.c model.c model_int8.c quant.c mathpin.c tokenizer.c json.c sha256.c
INT8_OBJS = $(INT8_SRCS:.c=.o)
INT8_BIN = cis2_verify3_int8

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

int8: $(INT8_BIN)

$(INT8_BIN): $(INT8_OBJS)
	$(CC) $(CFLAGS) -o $@ $(INT8_OBJS) $(LDFLAGS)

selftest: selftest.o mathpin.o tokenizer.o json.o sha256.o
	$(CC) $(CFLAGS) -o $@ selftest.o mathpin.o tokenizer.o json.o sha256.o $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Malformed-input regression: feeds tests/malformed/*.safetensors to
# cis2_model_load() in child processes; every file must be rejected with exit
# status 95 and no sanitizer report. Sanitizers are used when the compiler
# accepts them.
FUZZ_SRCS = tests/malformed_driver.c model.c mathpin.c json.c sha256.c
SAN_FLAGS := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -fsanitize=address,undefined -o /dev/null >/dev/null 2>&1 && echo -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)

fuzzcheck: $(FUZZ_SRCS) $(wildcard tests/malformed/*.safetensors)
	$(CC) $(CFLAGS) -g $(SAN_FLAGS) -o malformed_driver $(FUZZ_SRCS) $(LDFLAGS)
	./malformed_driver tests/malformed/*.safetensors

clean:
	rm -f $(OBJS) $(BIN) $(INT8_OBJS) $(INT8_BIN) selftest.o selftest malformed_driver

.PHONY: all clean int8 fuzzcheck
