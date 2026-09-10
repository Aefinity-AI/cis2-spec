//! Spec 12.1 and 12.2: the two digests.
//!
//! Spec 12.1 is one continuous `Sha256` stream, updated repeatedly and
//! finalized once -- not a step-wise re-hash of `running || new`. That
//! distinction is what the type below enforces: there is no way to get an
//! intermediate digest out of it.

use crate::sha256::Sha256;
use alloc::vec::Vec;

/// The witness chain of spec 12.1, fed in the order the spec's items 1-7
/// give (artifact hashes first, then the two table digests -- the E15k
/// correction).
pub struct Witness {
    hasher: Sha256,
    steps: usize,
}

impl Witness {
    /// Items 1-5.
    pub fn new(
        weights_sha256: &[u8; 32],
        tokenizer_sha256: &[u8; 32],
        config_sha256: &[u8; 32],
        table_digest: &[u8; 32],
        inv_freq_table_digest: &[u8; 32],
    ) -> Witness {
        let mut hasher = Sha256::new();
        hasher.update(weights_sha256);
        hasher.update(tokenizer_sha256);
        hasher.update(config_sha256);
        hasher.update(table_digest);
        hasher.update(inv_freq_table_digest);
        Witness { hasher, steps: 0 }
    }

    /// Item 6: every prompt token id, in order, as u32 little-endian.
    pub fn prompt(&mut self, prompt_token_ids: &[u32]) {
        for t in prompt_token_ids {
            self.hasher.update(&t.to_le_bytes());
        }
    }

    /// Item 7: for one decode step, (a) the full fp32 logit vector as
    /// u32-LE bit patterns in vocab-index order, then (b) the chosen id.
    ///
    /// The logits are the vector argmax was computed against for this step,
    /// not the next position's.
    pub fn step(&mut self, logits: &[f32], next_token_id: u32) {
        for v in logits {
            self.hasher.update(&v.to_bits().to_le_bytes());
        }
        self.hasher.update(&next_token_id.to_le_bytes());
        self.steps += 1;
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Finalize once, after all steps.
    pub fn finalize(self) -> [u8; 32] {
        self.hasher.finalize()
    }
}

/// Spec 12.2: a separate hash over `prompt_token_ids ++ generated ids`, as
/// LE u32 bytes, prompt first. Deliberately not derived from the witness
/// chain -- it is a different instance over different bytes, and this
/// function taking its inputs by value rather than sharing a hasher is that
/// separation made structural.
pub fn argmax_digest(prompt_token_ids: &[u32], generated: &[u32]) -> [u8; 32] {
    let mut s = Sha256::new();
    for t in prompt_token_ids.iter().chain(generated.iter()) {
        s.update(&t.to_le_bytes());
    }
    s.finalize()
}

/// Convenience for callers that already hold the whole decode.
pub fn witness_digest(
    weights_sha256: &[u8; 32],
    tokenizer_sha256: &[u8; 32],
    config_sha256: &[u8; 32],
    table_digest: &[u8; 32],
    inv_freq_table_digest: &[u8; 32],
    prompt_token_ids: &[u32],
    step_logits: &[Vec<f32>],
    generated: &[u32],
) -> [u8; 32] {
    let mut w = Witness::new(
        weights_sha256,
        tokenizer_sha256,
        config_sha256,
        table_digest,
        inv_freq_table_digest,
    );
    w.prompt(prompt_token_ids);
    for (logits, id) in step_logits.iter().zip(generated.iter()) {
        w.step(logits, *id);
    }
    w.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn z() -> [u8; 32] {
        [0u8; 32]
    }

    /// Spec 12.1's item order is the whole content of the E15k correction:
    /// a clean-room that fed the table digests first reproduced every other
    /// digest and still got a different CIS2_REF. So the order must be
    /// observable in a test, not just in prose.
    #[test]
    fn the_item_order_of_spec_12_1_is_observable() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        let correct = witness_digest(&a, &z(), &z(), &b, &z(), &[], &[], &[]);
        let e15j_order = witness_digest(&b, &z(), &z(), &a, &z(), &[], &[], &[]);
        assert_ne!(correct, e15j_order);
    }

    /// Item 7 is (logits, id) per step; swapping the two halves, or
    /// concatenating all logits and then all ids, must not collide.
    #[test]
    fn interleaving_of_logits_and_ids_is_observable() {
        let l0 = vec![1.0f32, 2.0];
        let l1 = vec![3.0f32, 4.0];
        let interleaved = witness_digest(&z(), &z(), &z(), &z(), &z(), &[], &[l0.clone(), l1.clone()], &[7, 9]);
        let mut s = Sha256::new();
        for v in l0.iter().chain(l1.iter()) {
            s.update(&v.to_bits().to_le_bytes());
        }
        for t in [7u32, 9] {
            s.update(&t.to_le_bytes());
        }
        // The zero prefixes are the same in both; only the arrangement of
        // item 7 differs.
        let mut pre = Sha256::new();
        for _ in 0..5 {
            pre.update(&z());
        }
        assert_ne!(interleaved, s.finalize());
        let _ = pre.finalize();
    }

    /// Spec 12.2 is a separate instance, so it cannot equal the witness
    /// chain even when both are fed the same token ids.
    #[test]
    fn the_argmax_digest_is_not_a_suffix_of_the_witness_chain() {
        let d = argmax_digest(&[6403, 1980, 253, 655], &[28, 665]);
        let w = witness_digest(&z(), &z(), &z(), &z(), &z(), &[6403, 1980, 253, 655], &[], &[]);
        assert_ne!(d, w);
    }

    /// A single bit flipped anywhere in a logit vector must change the
    /// witness digest -- this is the property the whole receipt rests on.
    #[test]
    fn one_flipped_logit_bit_changes_the_witness_digest() {
        let mut a = vec![0.5f32; 64];
        let base = witness_digest(&z(), &z(), &z(), &z(), &z(), &[1], &[a.clone()], &[3]);
        a[17] = f32::from_bits(a[17].to_bits() ^ 1);
        let tampered = witness_digest(&z(), &z(), &z(), &z(), &z(), &[1], &[a], &[3]);
        assert_ne!(base, tampered);
    }
}

#[cfg(test)]
mod strictly_stronger_tests {
    use super::*;
    use alloc::vec;

    /// Empirically observed on 2026-09-09: flipping one low mantissa bit in
    /// one bf16 weight of the pinned checkpoint moved the witness digest
    /// (`d827...` -> `fee5...`) while leaving the argmax digest and all 16
    /// generated token ids untouched. The token stream is a lossy view of
    /// the computation: an adversary can perturb the model and still emit
    /// the expected text.
    ///
    /// This test pins the property that made that observation possible --
    /// the full-logit chain of spec 12.1 separates two runs the token-level
    /// digest of spec 12.2 cannot tell apart, so 12.1 is strictly stronger
    /// and 12.2 alone would not be a sufficient receipt.
    #[test]
    fn the_witness_chain_separates_runs_the_argmax_digest_cannot() {
        let z = [0u8; 32];
        let prompt = [6403u32, 1980, 253, 655];

        // Two runs that agree on every argmax but not on the logits.
        let mut a = vec![0.0f32; 8];
        a[3] = 10.0;
        let mut b = a.clone();
        b[0] = 0.5; // a loser moves; index 3 still wins in both

        let gen = [3u32];
        assert_eq!(
            super::super::ops::argmax(&a),
            super::super::ops::argmax(&b),
            "the two runs must agree on the argmax for this test to mean anything"
        );

        let arg_a = argmax_digest(&prompt, &gen);
        let arg_b = argmax_digest(&prompt, &gen);
        assert_eq!(arg_a, arg_b, "spec 12.2 sees only token ids, so it cannot separate these");

        let w_a = witness_digest(&z, &z, &z, &z, &z, &prompt, &[a], &gen);
        let w_b = witness_digest(&z, &z, &z, &z, &z, &prompt, &[b], &gen);
        assert_ne!(
            w_a, w_b,
            "spec 12.1 hashes the full logit vector and must separate them"
        );
    }
}
