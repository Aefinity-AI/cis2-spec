//! Spec 2.5 / spec 4: the safetensors container, and the bf16 -> fp32
//! widening applied to every tensor as it is loaded.
//!
//! The container format is not defined by CIS-2; it is the file format the
//! pinned artifact happens to be in. What CIS-2 does pin is what comes out
//! of it: row-major `[out_features, in_features]` tensors, every element a
//! bf16 read as `u16::from_le_bytes`, widened by the exact bit-shift of
//! spec 4.1 and never used in bf16 form (spec 4.2).

use crate::json::{parse, Json};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Spec 4.1, verbatim: an exact, lossless bit-shift, not a rounding
/// conversion and not a call through any library "convert" routine.
#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// The widened elements, in the file's row-major order (spec 2.5).
    pub data: Vec<f32>,
}

impl Tensor {
    /// Row `r` of a 2-D `[out_features, in_features]` tensor.
    pub fn row(&self, r: usize, in_features: usize) -> &[f32] {
        &self.data[r * in_features..(r + 1) * in_features]
    }
}

pub struct SafeTensors {
    tensors: Vec<Tensor>,
}

impl SafeTensors {
    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| alloc::format!("safetensors: no tensor named {name:?}"))
    }

    /// Fetch and check the shape in one step, so a mis-shaped checkpoint
    /// fails at load with a readable message instead of at the first
    /// out-of-bounds slice.
    pub fn get_2d(&self, name: &str, rows: usize, cols: usize) -> Result<&Tensor, String> {
        let t = self.get(name)?;
        if t.shape != [rows, cols] {
            return Err(alloc::format!(
                "safetensors: {name} has shape {:?}, spec 2.5 requires [{rows}, {cols}]",
                t.shape
            ));
        }
        Ok(t)
    }

    pub fn get_1d(&self, name: &str, n: usize) -> Result<&Tensor, String> {
        let t = self.get(name)?;
        if t.shape != [n] {
            return Err(alloc::format!(
                "safetensors: {name} has shape {:?}, spec 2.5 requires [{n}]",
                t.shape
            ));
        }
        Ok(t)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }
}

pub fn load(file: &[u8]) -> Result<SafeTensors, String> {
    if file.len() < 8 {
        return Err("safetensors: file is shorter than its 8-byte header length".into());
    }
    let mut n_bytes = [0u8; 8];
    n_bytes.copy_from_slice(&file[0..8]);
    let header_len = u64::from_le_bytes(n_bytes);
    // Bound the header before using it as a length: a corrupt or hostile
    // file must produce an error, never a panicking slice.
    let header_len: usize = header_len
        .try_into()
        .map_err(|_| "safetensors: header length does not fit in a usize")?;
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or("safetensors: header length overflows the file offset")?;
    if data_start > file.len() {
        return Err("safetensors: header length runs past the end of the file".into());
    }
    let header = parse(&file[8..data_start]).map_err(|e| alloc::format!("safetensors header: {e}"))?;
    let fields = match &header {
        Json::Obj(fields) => fields,
        _ => return Err("safetensors: header is not a JSON object".into()),
    };
    let payload = &file[data_start..];

    let mut tensors: Vec<Tensor> = Vec::new();
    for (name, entry) in fields {
        // `__metadata__` carries free-form strings, not a tensor.
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| alloc::format!("safetensors: {name} has no dtype"))?;
        // Spec 2.5: every tensor in this checkpoint is BF16, and spec 4 only
        // defines widening from BF16. Refuse anything else rather than
        // reinterpret its bytes.
        if dtype != "BF16" {
            return Err(alloc::format!(
                "safetensors: {name} has dtype {dtype}; spec 2.5 requires BF16 for every tensor"
            ));
        }
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(|v| v.as_arr())
            .ok_or_else(|| alloc::format!("safetensors: {name} has no shape"))?
            .iter()
            .map(|d| d.as_usize().ok_or("safetensors: non-integer dimension"))
            .collect::<Result<_, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_arr())
            .ok_or_else(|| alloc::format!("safetensors: {name} has no data_offsets"))?;
        if offsets.len() != 2 {
            return Err(alloc::format!("safetensors: {name} has malformed data_offsets"));
        }
        let begin = offsets[0]
            .as_usize()
            .ok_or("safetensors: non-integer data offset")?;
        let end = offsets[1]
            .as_usize()
            .ok_or("safetensors: non-integer data offset")?;
        if end < begin || end > payload.len() {
            return Err(alloc::format!(
                "safetensors: {name}'s data_offsets [{begin}, {end}] are outside the payload"
            ));
        }
        let n_elems = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| alloc::format!("safetensors: {name}'s shape overflows"))?;
        if end - begin != n_elems * 2 {
            return Err(alloc::format!(
                "safetensors: {name} spans {} bytes, but its shape needs {} BF16 elements",
                end - begin,
                n_elems
            ));
        }
        let raw = &payload[begin..end];
        // Spec 4.1/4.2: widen every element here, at load time, so no
        // arithmetic anywhere downstream ever sees a bf16.
        let mut data = vec![0.0f32; n_elems];
        for (i, out) in data.iter_mut().enumerate() {
            let b = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            *out = bf16_to_f32(b);
        }
        tensors.push(Tensor {
            name: name.clone(),
            shape,
            data,
        });
    }
    Ok(SafeTensors { tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec 4.1 is a bit-shift, so these identities are exact, not
    /// approximate: the widened value's low 16 mantissa bits are zero and
    /// its top 16 bits are the original bf16 pattern.
    #[test]
    fn widening_is_the_exact_bit_shift_spec_4_1_writes() {
        assert_eq!(bf16_to_f32(0x3F80).to_bits(), 0x3F80_0000); // 1.0
        assert_eq!(bf16_to_f32(0xBF80).to_bits(), 0xBF80_0000); // -1.0
        assert_eq!(bf16_to_f32(0x0000).to_bits(), 0x0000_0000); // +0
        assert_eq!(bf16_to_f32(0x8000).to_bits(), 0x8000_0000); // -0
        assert_eq!(bf16_to_f32(0x7F80).to_bits(), 0x7F80_0000); // +inf
        for b in 0u16..=u16::MAX {
            assert_eq!(bf16_to_f32(b).to_bits() >> 16, b as u32);
            assert_eq!(bf16_to_f32(b).to_bits() & 0xFFFF, 0);
        }
    }

    fn tiny_file() -> Vec<u8> {
        let header = br#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(header.len() as u64).to_le_bytes());
        f.extend_from_slice(header);
        for b in [0x3F80u16, 0xC000, 0x0000, 0x4040] {
            f.extend_from_slice(&b.to_le_bytes());
        }
        f
    }

    #[test]
    fn loads_a_tensor_in_row_major_order() {
        let st = load(&tiny_file()).unwrap();
        let t = st.get_2d("a", 2, 2).unwrap();
        assert_eq!(t.data, vec![1.0f32, -2.0, 0.0, 3.0]);
        assert_eq!(t.row(1, 2), &[0.0f32, 3.0]);
    }

    #[test]
    fn truncated_or_lying_headers_error_rather_than_panic() {
        let good = tiny_file();
        // A header length longer than the file.
        let mut bad = good.clone();
        bad[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(load(&bad).is_err());
        // Offsets that claim more bytes than the shape needs.
        let header = br#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut bad = Vec::new();
        bad.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bad.extend_from_slice(header);
        bad.extend_from_slice(&[0u8; 16]);
        assert!(load(&bad).is_err());
        // Everything after the 8-byte prefix removed.
        assert!(load(&good[..8]).is_err());
        assert!(load(&[]).is_err());
    }

    #[test]
    fn a_non_bf16_tensor_is_refused_rather_than_reinterpreted() {
        let header = br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(header.len() as u64).to_le_bytes());
        f.extend_from_slice(header);
        f.extend_from_slice(&[0u8; 2]);
        let err = match load(&f) {
            Ok(_) => panic!("a non-BF16 tensor was accepted"),
            Err(e) => e,
        };
        assert!(err.contains("BF16"), "{err}");
    }
}
