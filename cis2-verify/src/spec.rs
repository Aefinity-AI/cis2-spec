//! The constants CIS-2 v0.3b pins, in one place, each with its section
//! number. Nothing else in this crate hardcodes a digest or a token id.

/// Spec 2.1 artifact hashes for the pinned SmolLM2-135M checkpoint.
pub const WEIGHTS_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";
pub const CONFIG_SHA256: &str =
    "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843";
pub const TOKENIZER_SHA256: &str =
    "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c";

/// Spec 3.2 / 3.3.
pub const PROMPT: &str = "Once upon a time";
pub const PROMPT_TOKEN_IDS: [u32; 4] = [6403, 1980, 253, 655];
/// Spec 3.4: exactly 16, EOS never checked.
pub const GEN_TOKS: usize = 16;

/// Spec 6.6.
pub const TABLE_DIGEST: &str =
    "23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d";
/// Spec 7.2.
pub const INV_FREQ_TABLE_DIGEST: &str =
    "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12";
/// Spec 12.1.
pub const CIS2_REF: &str = "d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df";
/// Spec 12.2.
pub const ARGMAX_DIGEST: &str =
    "0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff";
/// Spec 13.1.
pub const GENERATED_TOKEN_IDS: [u32; 16] = [
    28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351,
];
