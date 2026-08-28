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

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

selftest: selftest.o mathpin.o tokenizer.o json.o sha256.o
	$(CC) $(CFLAGS) -o $@ selftest.o mathpin.o tokenizer.o json.o sha256.o $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(BIN) selftest.o selftest

.PHONY: all clean
