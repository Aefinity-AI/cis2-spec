//! `cis2-verify` --- a standalone verifier for CIS-2 v0.3b.
//!
//! Written from `CIS2_SPEC_v0.3b.md` alone, with no dependency on any
//! crate and no code shared with the reference implementation. See
//! `CLEANROOM.md` for what that boundary means and what it does not.
//!
//! The claim this crate is evidence for is narrow and worth stating
//! precisely: that the specification is *sufficient* --- that an
//! implementation written from the document reproduces the pinned digests
//! bit for bit. It is not a claim that this implementation is fast, that
//! it is the only conforming one, or that bit-identical inference is new.
//! Prior and concurrent work on reproducible floating point and on
//! deterministic inference is cited in the specification's own references.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod check;
pub mod config;
pub mod fpenv;
pub mod hex;
pub mod json;
pub mod mathpin;
pub mod model;
pub mod ops;
pub mod receipt;
pub mod safetensors;
pub mod sha256;
pub mod softfp;
pub mod spec;
pub mod tokenizer;
pub mod unicode;
pub mod verify;
pub mod witness;
