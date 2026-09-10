//! A minimal JSON reader, written for this crate so that parsing
//! `config.json`, `tokenizer.json` and the safetensors header pulls in no
//! dependency (see `CLEANROOM.md`).
//!
//! This is deliberately not a general-purpose JSON library. It accepts the
//! subset RFC 8259 defines and that these three artifacts use, and it
//! rejects everything else with a static error string rather than guessing.
//! The one place where it must be exactly right is number parsing, because
//! `rms_norm_eps` becomes an f32 the spec pins bit-for-bit (spec 2.3,
//! `EPS_F32 = 0x3727C5AC`); `Json::as_f32` documents how that is reached and
//! `config.rs` asserts the resulting bit pattern.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The number's value, plus the exact source text it was written as.
    ///
    /// The value is `None` when the literal is outside the range this
    /// parser can convert with a single correct rounding (more than 16
    /// significant digits, or a decimal exponent past +/-22). That is not
    /// an error at parse time, because artifacts carry fields nobody in
    /// this crate reads -- SmolLM2-135M's `config.json` writes
    /// `initializer_range` with 17 digits -- but it does mean a caller that
    /// asks for that particular field's value gets `None` rather than an
    /// approximation. Keeping the source text lets callers that need an
    /// integer refuse a value written with a fraction or an exponent
    /// instead of silently truncating it.
    Num(Option<f64>, String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// An integer, only if it was *written* as one. `576` yields
    /// `Some(576)`; `576.0`, `5.76e2` and `-1` all yield `None` for a
    /// `usize` request. A config that writes a dimension as a float is a
    /// config this verifier refuses rather than rounds.
    pub fn as_u64(&self) -> Option<u64> {
        let text = match self {
            Json::Num(_, text) => text.as_str(),
            _ => return None,
        };
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut acc: u64 = 0;
        for b in text.bytes() {
            acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
        }
        Some(acc)
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|v| v as usize)
    }

    /// The f32 the number denotes, RNE-rounded once from the exactly
    /// computed f64 value.
    ///
    /// `parse_number` builds the f64 as `significand * 10^k` using only
    /// exactly-representable operands (every `10^k` for `|k| <= 22` is
    /// exact in f64, and a significand of at most 15 digits is exact), so
    /// that f64 is the correctly-rounded value of the literal. Both
    /// `rope_theta = 100000.0` and `rms_norm_eps = 1e-05` are in that
    /// range. Anything outside it is rejected at parse time rather than
    /// approximated, so this cast is a single rounding step.
    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v, _) => *v,
            _ => None,
        }
    }

    /// The literal as written, for a number; used to report a field this
    /// parser declined to convert.
    pub fn num_text(&self) -> Option<&str> {
        match self {
            Json::Num(_, text) => Some(text.as_str()),
            _ => None,
        }
    }
}

pub fn parse(input: &[u8]) -> Result<Json, &'static str> {
    let mut p = Parser { b: input, i: 0 };
    p.skip_ws();
    let v = p.value(0)?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err("json: trailing bytes after the top-level value");
    }
    Ok(v)
}

/// Nesting cap. `tokenizer.json` reaches depth 4; a hostile file that
/// nests a million arrays would otherwise recurse this parser off its
/// stack, which is a crash, not a verdict.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            match self.b[self.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), &'static str> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err("json: expected a different delimiter")
        }
    }

    fn literal(&mut self, word: &[u8]) -> Result<(), &'static str> {
        if self.b.len() - self.i >= word.len() && &self.b[self.i..self.i + word.len()] == word {
            self.i += word.len();
            Ok(())
        } else {
            Err("json: bad literal")
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, &'static str> {
        if depth > MAX_DEPTH {
            return Err("json: nesting deeper than this parser accepts");
        }
        match self.peek().ok_or("json: unexpected end of input")? {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => {
                self.literal(b"true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.literal(b"false")?;
                Ok(Json::Bool(false))
            }
            b'n' => {
                self.literal(b"null")?;
                Ok(Json::Null)
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err("json: unexpected byte where a value was expected"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, &'static str> {
        self.eat(b'{')?;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            let val = self.value(depth + 1)?;
            fields.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                _ => return Err("json: expected ',' or '}' in object"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, &'static str> {
        self.eat(b'[')?;
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err("json: expected ',' or ']' in array"),
            }
        }
    }

    fn string(&mut self) -> Result<String, &'static str> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = *self.b.get(self.i).ok_or("json: unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("json: unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // Surrogate pair: JSON encodes a codepoint above
                            // the BMP as two \u escapes, and the tokenizer's
                            // vocabulary is allowed to contain such bytes.
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                if self.peek() != Some(b'\\') {
                                    return Err("json: lone high surrogate");
                                }
                                self.i += 1;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("json: high surrogate not followed by a low one");
                                }
                                let cp = 0x1_0000u32
                                    + (((hi - 0xD800) as u32) << 10)
                                    + (lo - 0xDC00) as u32;
                                char::from_u32(cp).ok_or("json: bad surrogate pair")?
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err("json: lone low surrogate");
                            } else {
                                char::from_u32(hi as u32).ok_or("json: bad \\u escape")?
                            };
                            out.push(ch);
                        }
                        _ => return Err("json: unknown escape"),
                    }
                }
                // A raw control byte is invalid JSON; anything else is copied
                // through, and multi-byte UTF-8 sequences fall out of this
                // byte-at-a-time copy correctly because their continuation
                // bytes are all >= 0x80.
                0x00..=0x1F => return Err("json: raw control character in string"),
                _ => {
                    let start = self.i - 1;
                    let len = utf8_len(c)?;
                    if self.b.len() < start + len {
                        return Err("json: truncated UTF-8 sequence");
                    }
                    let s = core::str::from_utf8(&self.b[start..start + len])
                        .map_err(|_| "json: invalid UTF-8 in string")?;
                    out.push_str(s);
                    self.i = start + len;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u16, &'static str> {
        if self.b.len() < self.i + 4 {
            return Err("json: truncated \\u escape");
        }
        let mut v: u16 = 0;
        for k in 0..4 {
            let d = self.b[self.i + k];
            let n = match d {
                b'0'..=b'9' => d - b'0',
                b'a'..=b'f' => d - b'a' + 10,
                b'A'..=b'F' => d - b'A' + 10,
                _ => return Err("json: non-hex digit in \\u escape"),
            };
            v = (v << 4) | n as u16;
        }
        self.i += 4;
        Ok(v)
    }

    /// Number, built exactly. See `Json::as_f32`.
    fn number(&mut self) -> Result<Json, &'static str> {
        let start = self.i;
        let neg = if self.peek() == Some(b'-') {
            self.i += 1;
            true
        } else {
            false
        };
        // Integer part.
        let int_start = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.i == int_start {
            return Err("json: number with no integer part");
        }
        if self.i - int_start > 1 && self.b[int_start] == b'0' {
            return Err("json: number with a leading zero");
        }
        // 15 significant decimal digits is the most that is guaranteed to
        // round-trip through f64; refusing a longer literal is better than
        // returning a value that is one ULP off what the file says.
        let mut sig: u64 = 0;
        let mut digits = 0usize;
        let mut exp10: i32 = 0;
        // `inexact` marks a literal this parser will not convert: see
        // `Json::Num`. Scanning continues either way so the parse position
        // stays correct and the rest of the document still loads.
        let mut inexact = false;
        for k in int_start..self.i {
            let d = (self.b[k] - b'0') as u64;
            if digits == 0 && d == 0 {
                continue; // leading zero contributes nothing
            }
            if digits >= 16 {
                inexact = true;
                exp10 += 1;
                continue;
            }
            sig = sig * 10 + d;
            digits += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            let frac_start = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == frac_start {
                return Err("json: number with a '.' and no fraction digits");
            }
            for k in frac_start..self.i {
                let d = (self.b[k] - b'0') as u64;
                if digits == 0 && d == 0 {
                    exp10 -= 1;
                    continue;
                }
                if digits >= 16 {
                    inexact = true;
                    continue;
                }
                sig = sig * 10 + d;
                digits += 1;
                exp10 -= 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;
            let esign = match self.peek() {
                Some(b'+') => {
                    self.i += 1;
                    1i32
                }
                Some(b'-') => {
                    self.i += 1;
                    -1i32
                }
                _ => 1i32,
            };
            let e_start = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == e_start {
                return Err("json: exponent with no digits");
            }
            let mut e: i32 = 0;
            for k in e_start..self.i {
                e = e
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((self.b[k] - b'0') as i32))
                    .ok_or("json: exponent out of range")?;
                if e > 10_000 {
                    return Err("json: exponent out of range");
                }
            }
            exp10 += esign * e;
        }
        let text = core::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "json: non-ASCII in number")?;

        // `sig` carries at most 16 digits, so it is below 2^53 and exact in
        // f64; every `10^k` for `|k| <= 22` is exact too. One multiply or
        // divide then puts the result a single correct rounding away from
        // the literal. Outside that envelope the value is left unconverted
        // rather than approximated.
        let value = if inexact || !(-22..=22).contains(&exp10) {
            None
        } else if sig == 0 {
            Some(if neg { -0.0f64 } else { 0.0f64 })
        } else {
            let p = POW10[exp10.unsigned_abs() as usize];
            let v = if exp10 >= 0 {
                (sig as f64) * p
            } else {
                (sig as f64) / p
            };
            Some(if neg { -v } else { v })
        };
        Ok(Json::Num(value, String::from(text)))
    }
}

fn utf8_len(lead: u8) -> Result<usize, &'static str> {
    match lead {
        0x00..=0x7F => Ok(1),
        0xC2..=0xDF => Ok(2),
        0xE0..=0xEF => Ok(3),
        0xF0..=0xF4 => Ok(4),
        _ => Err("json: invalid UTF-8 lead byte"),
    }
}

/// `10^0 .. 10^22`, the range in which every power of ten is exactly
/// representable in binary64.
static POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_the_artifacts_use() {
        let v = parse(br#"{"a":[1,2,3],"b":"x","c":true,"d":null,"e":{"f":-2.5}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap().len(), 3);
        assert_eq!(v.get("b").unwrap().as_str(), Some("x"));
        assert_eq!(v.get("c").unwrap().as_bool(), Some(true));
        assert_eq!(*v.get("d").unwrap(), Json::Null);
        assert_eq!(v.get("e").unwrap().get("f").unwrap().as_f64(), Some(-2.5));
    }

    /// The one number in the whole pipeline whose bits the spec pins.
    #[test]
    fn rms_norm_eps_literal_lands_on_the_pinned_bit_pattern() {
        let v = parse(br#"{"rms_norm_eps":1e-05}"#).unwrap();
        assert_eq!(
            v.get("rms_norm_eps").unwrap().as_f32().unwrap().to_bits(),
            0x3727C5AC
        );
    }

    #[test]
    fn integers_written_as_floats_are_refused() {
        let v = parse(br#"{"hidden_size":576.0}"#).unwrap();
        assert_eq!(v.get("hidden_size").unwrap().as_usize(), None);
        let v = parse(br#"{"hidden_size":576}"#).unwrap();
        assert_eq!(v.get("hidden_size").unwrap().as_usize(), Some(576));
    }

    #[test]
    fn rope_theta_parses_exactly() {
        let v = parse(br#"{"rope_theta":100000.0}"#).unwrap();
        assert_eq!(v.get("rope_theta").unwrap().as_f32(), Some(100000.0f32));
    }

    #[test]
    fn escapes_and_surrogate_pairs_round_trip() {
        // The escaped forms, which is how the vocabulary in tokenizer.json
        // may spell a non-ASCII token, and the literal UTF-8 form.
        let v = parse(br#""a\u0120b\n\ud83d\ude00""#).unwrap();
        assert_eq!(v.as_str(), Some("a\u{0120}b\n\u{1F600}"));
        let v = parse("\"a\u{0120}b\u{1F600}\"".as_bytes()).unwrap();
        assert_eq!(v.as_str(), Some("a\u{0120}b\u{1F600}"));
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        for bad in [
            &b"{"[..],
            b"[1,]",
            b"{\"a\" 1}",
            b"\"unterminated",
            b"01",
            b"tru",
            b"{}x",
            b"1e99999999999999999999",
            b"1e+",
        ] {
            assert!(parse(bad).is_err(), "accepted {bad:?}");
        }
    }
}

#[cfg(test)]
mod inexact_tests {
    use super::*;

    /// SmolLM2-135M's `config.json` carries `initializer_range` with 17
    /// significant digits. This crate never reads that field, so the file
    /// must still load -- but asking for its value must give `None` rather
    /// than a silently rounded number.
    #[test]
    fn a_field_this_parser_will_not_convert_does_not_stop_the_document() {
        let v = parse(br#"{"initializer_range":0.041666666666666664,"hidden_size":576}"#).unwrap();
        assert_eq!(v.get("hidden_size").unwrap().as_usize(), Some(576));
        assert_eq!(v.get("initializer_range").unwrap().as_f64(), None);
        assert_eq!(
            v.get("initializer_range").unwrap().num_text(),
            Some("0.041666666666666664")
        );
    }

    /// `rope_theta` is written as a bare integer in the pinned config, not
    /// as `100000.0`; both spellings must reach the same f32.
    #[test]
    fn rope_theta_written_as_an_integer_is_the_same_value() {
        let a = parse(br#"{"rope_theta":100000}"#).unwrap();
        let b = parse(br#"{"rope_theta":100000.0}"#).unwrap();
        assert_eq!(a.get("rope_theta").unwrap().as_f32(), Some(100000.0f32));
        assert_eq!(
            a.get("rope_theta").unwrap().as_f32(),
            b.get("rope_theta").unwrap().as_f32()
        );
    }
}
