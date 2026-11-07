//! Minimal torch `.pth` (zip-of-pickle) reader — first written for the native Wan converter
//! (sc-3237 / sc-3224), shared from mlx-gen since sc-24833 (the E-LatentLPIPS checkpoints ship as
//! `torch.save` state dicts too, BatchNorm `num_batches_tracked` int64 counters included).
//!
//! Native Wan checkpoints ship the T5 encoder (`models_t5_umt5-xxl-enc-bf16.pth`) and the VAE
//! (`Wan2.x_VAE.pth`) as PyTorch `torch.save` archives — a ZIP holding `<prefix>/data.pkl` (a
//! pickled `OrderedDict` whose tensors are `persistent_id` storage references) plus `<prefix>/data/<n>`
//! raw storage blobs. The reference Python loads these with `torch.load(...).float()` (every tensor
//! → f32), so [`load_pth_f32`] mirrors that exactly: it returns each tensor as an **f32** MLX
//! [`Array`] in PyTorch layout (the converter's sanitizers then transpose conv weights to
//! channels-last and cast per component).
//!
//! Scope is deliberately narrow — exactly the opcode set `torch.save` emits (protocol 2/4, STORED
//! zip entries) and the three globals it references (`collections.OrderedDict`,
//! `torch.FloatStorage`/`BFloat16Storage`/`HalfStorage`, `torch._utils._rebuild_tensor_v2`). The
//! `zip` crate handles the archive (incl. data-descriptor entries and zip64 for the >4 GB T5).
//! Vendored rather than pulling in candle-core (a whole second tensor framework) for one file read.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::{Error, Result};
use mlx_rs::Array;

/// Torch storage element type (the `torch.<X>Storage` global in the pickle). The float storages the
/// Wan T5/VAE use, plus `LongStorage` (BatchNorm's `num_batches_tracked` counter in any module
/// state dict with a BatchNorm); everything is decoded to f32 (mirroring `.float()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageDtype {
    Float32,
    BFloat16,
    Float16,
    Int64,
}

impl StorageDtype {
    fn from_global(name: &str) -> Result<Self> {
        match name {
            "torch FloatStorage" => Ok(StorageDtype::Float32),
            "torch BFloat16Storage" => Ok(StorageDtype::BFloat16),
            "torch HalfStorage" => Ok(StorageDtype::Float16),
            "torch LongStorage" => Ok(StorageDtype::Int64),
            other => Err(Error::Msg(format!(
                "unsupported torch storage type `{other}` in .pth (expected Float/BFloat16/Half/Long)"
            ))),
        }
    }

    fn elem_size(self) -> usize {
        match self {
            StorageDtype::Float32 => 4,
            StorageDtype::BFloat16 | StorageDtype::Float16 => 2,
            StorageDtype::Int64 => 8,
        }
    }
}

/// A pickle stack value. Only the variants `torch.save` produces are modeled.
#[derive(Clone, Debug)]
enum Val {
    Int(i64),
    Str(String),
    /// `requires_grad` (NEWTRUE/NEWFALSE) — a stack placeholder; the value is never read.
    Bool,
    None,
    Mark,
    Global(String),
    Tuple(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    /// A resolved `persistent_id` storage reference: `('storage', <dtype>, <key>, <loc>, <numel>)`.
    Storage {
        dtype: StorageDtype,
        key: String,
    },
    /// The result of `_rebuild_tensor_v2(storage, offset, size, stride, …)`.
    Tensor {
        dtype: StorageDtype,
        key: String,
        offset: i64,
        size: Vec<i64>,
        stride: Vec<i64>,
    },
}

/// Upper bound on opcodes executed by [`parse_pickle`]. A `data.pkl` from even a large checkpoint
/// is a flat `{name: tensor}` dict — millions of opcodes is already far past any legitimate file,
/// so this caps a stream that loops or never reaches `STOP`.
const MAX_PICKLE_OPCODES: u64 = 64 * 1024 * 1024;

/// Cumulative budget on [`Val`] nodes *materialized by clones* (memoize / fetch). The memo clones
/// values in and out by deep copy, and `TUPLE*` nest them, so a crafted pickle can `BINGET` a value,
/// pair it with `TUPLE2`, and `BINPUT` the result — doubling the in-memory structure every ~6 bytes
/// (a "billion-laughs" amplification) and demanding `2^O(n)` memory before any tensor validation runs.
/// Charging each clone against this budget — *before* the clone is made — bounds both the resident
/// memory and the total work to `O(budget)`. A real state dict materializes well under a million such
/// nodes, so 8M is ~8× headroom while still tripping the amplification within a couple dozen doublings.
const MAX_PICKLE_NODES: u64 = 8 * 1024 * 1024;

/// Count the [`Val`] nodes in a value (itself plus every nested element), used to charge clones
/// against [`MAX_PICKLE_NODES`].
fn count_nodes(v: &Val) -> u64 {
    match v {
        Val::Tuple(items) => 1 + items.iter().map(count_nodes).sum::<u64>(),
        Val::Dict(items) => {
            1 + items
                .iter()
                .map(|(k, val)| count_nodes(k) + count_nodes(val))
                .sum::<u64>()
        }
        _ => 1,
    }
}

/// Add `n` to the running clone budget, erroring once it crosses [`MAX_PICKLE_NODES`].
fn charge_nodes(total: &mut u64, n: u64) -> Result<()> {
    *total = total.saturating_add(n);
    if *total > MAX_PICKLE_NODES {
        return Err(Error::Msg(format!(
            "pickle: node budget exceeded ({MAX_PICKLE_NODES}); refusing to parse (possible memory-amplification attack)"
        )));
    }
    Ok(())
}

/// A cursor over the pickle byte stream.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| Error::Msg("pickle: EOF".into()))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| Error::Msg("pickle: read past EOF".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u16le(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32le(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    /// Read a `\n`-terminated line (for the `GLOBAL` opcode's module / name).
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while *self
            .buf
            .get(self.pos)
            .ok_or_else(|| Error::Msg("pickle: unterminated line".into()))?
            != b'\n'
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos])
            .map_err(|e| Error::Msg(format!("pickle: bad utf8 in global: {e}")))?
            .to_string();
        self.pos += 1; // skip '\n'
        Ok(s)
    }
}

/// Run the pickle VM over `data.pkl`, returning the `(name, Tensor-spec)` entries of the top dict.
fn parse_pickle(data: &[u8]) -> Result<Vec<(String, Val)>> {
    let mut r = Reader { buf: data, pos: 0 };
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut opcodes: u64 = 0;
    // Total Val nodes cloned through the memo so far (billion-laughs guard, F-015).
    let mut cloned_nodes: u64 = 0;

    let pop = |st: &mut Vec<Val>| {
        st.pop()
            .ok_or_else(|| Error::Msg("pickle: stack underflow".into()))
    };
    // Collect items back to the topmost Mark (exclusive), restoring original order.
    let pop_to_mark = |st: &mut Vec<Val>| -> Result<Vec<Val>> {
        let mut items = Vec::new();
        loop {
            match st.pop() {
                Some(Val::Mark) => break,
                Some(v) => items.push(v),
                None => return Err(Error::Msg("pickle: no Mark on stack".into())),
            }
        }
        items.reverse();
        Ok(items)
    };

    loop {
        opcodes += 1;
        if opcodes > MAX_PICKLE_OPCODES {
            return Err(Error::Msg(format!(
                "pickle: opcode budget exceeded ({MAX_PICKLE_OPCODES}); refusing to parse"
            )));
        }
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?;
            } // PROTO
            0x95 => {
                r.take(8)?;
            } // FRAME
            b'.' => break, // STOP
            b'c' => {
                // GLOBAL module '\n' name '\n'
                let module = r.line()?;
                let name = r.line()?;
                stack.push(Val::Global(format!("{module} {name}")));
            }
            0x93 => {
                // STACK_GLOBAL: name, module on stack
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                if let (Val::Str(m), Val::Str(n)) = (module, name) {
                    stack.push(Val::Global(format!("{m} {n}")));
                } else {
                    return Err(Error::Msg("pickle: STACK_GLOBAL non-string".into()));
                }
            }
            b'}' => stack.push(Val::Dict(Vec::new())), // EMPTY_DICT
            b')' => stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
            b']' => stack.push(Val::Tuple(Vec::new())), // EMPTY_LIST (modeled as tuple; unused)
            b'(' => stack.push(Val::Mark),             // MARK
            b'N' => stack.push(Val::None),             // NONE
            0x88 => stack.push(Val::Bool),             // NEWTRUE
            0x89 => stack.push(Val::Bool),             // NEWFALSE
            b'J' => stack.push(Val::Int(r.i32le()? as i64)), // BININT
            b'K' => stack.push(Val::Int(r.u8()? as i64)), // BININT1
            b'M' => stack.push(Val::Int(r.u16le()? as i64)), // BININT2
            0x8a => {
                // LONG1: 1-byte length, then little-endian signed
                let n = r.u8()? as usize;
                // An i64 holds at most 8 bytes; torch never emits a wider int for the metadata this
                // VM reads. Reject n > 8 — otherwise `(b as i64) << (8 * i)` shifts by >= 64, which
                // panics in debug and silently masks to a wrong value in release (F-016).
                if n > 8 {
                    return Err(Error::Msg(format!(
                        "pickle: LONG1 length {n} exceeds 8 bytes (i64)"
                    )));
                }
                let bytes = r.take(n)?;
                let mut v: i64 = 0;
                for (i, &b) in bytes.iter().enumerate() {
                    v |= (b as i64) << (8 * i);
                }
                if n > 0 && bytes[n - 1] & 0x80 != 0 && n < 8 {
                    v -= 1i64 << (8 * n); // sign-extend
                }
                stack.push(Val::Int(v));
            }
            b'X' => {
                // BINUNICODE: u32 length + utf8
                let n = r.u32le()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            0x8c => {
                // SHORT_BINUNICODE: u8 length + utf8
                let n = r.u8()? as usize;
                let s = std::str::from_utf8(r.take(n)?)
                    .map_err(|e| Error::Msg(format!("pickle: bad utf8 string: {e}")))?
                    .to_string();
                stack.push(Val::Str(s));
            }
            b'q' => {
                let i = r.u8()? as u32;
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Msg("pickle: BINPUT empty".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(top))?;
                memo.insert(i, top.clone());
            }
            b'r' => {
                let i = r.u32le()?;
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Msg("pickle: LONG_BINPUT empty".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(top))?;
                memo.insert(i, top.clone());
            }
            b'h' => {
                let i = r.u8()? as u32;
                let v = memo
                    .get(&i)
                    .ok_or_else(|| Error::Msg("pickle: BINGET miss".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(v))?;
                stack.push(v.clone());
            }
            b'j' => {
                let i = r.u32le()?;
                let v = memo
                    .get(&i)
                    .ok_or_else(|| Error::Msg("pickle: LONG_BINGET miss".into()))?;
                charge_nodes(&mut cloned_nodes, count_nodes(v))?;
                stack.push(v.clone());
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a]));
            } // TUPLE1
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b]));
            } // TUPLE2
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b, c]));
            } // TUPLE3
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            } // TUPLE
            b'Q' => {
                // BINPERSID: pop the storage tuple, resolve.
                let pid = pop(&mut stack)?;
                stack.push(resolve_persid(pid)?);
            }
            b'R' => {
                // REDUCE: callable, args → result
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(apply_reduce(callable, args)?);
            }
            b's' => {
                // SETITEM: dict, key, value
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(items)) => items.push((key, value)),
                    _ => return Err(Error::Msg("pickle: SETITEM target not a dict".into())),
                }
            }
            b'u' => {
                // SETITEMS: dict, MARK, k1, v1, … → dict
                let pairs = pop_to_mark(&mut stack)?;
                if pairs.len() % 2 != 0 {
                    return Err(Error::Msg("pickle: SETITEMS odd count".into()));
                }
                match stack.last_mut() {
                    Some(Val::Dict(items)) => {
                        for kv in pairs.chunks_exact(2) {
                            items.push((kv[0].clone(), kv[1].clone()));
                        }
                    }
                    _ => return Err(Error::Msg("pickle: SETITEMS target not a dict".into())),
                }
            }
            b'b' => {
                // BUILD: obj, state → obj  (we discard state: OrderedDict items arrive via SETITEMS,
                // state is the empty instance __dict__).
                let _state = pop(&mut stack)?;
            }
            other => {
                return Err(Error::Msg(format!(
                    "pickle: unsupported opcode 0x{other:02x} at byte {} (torch.save surface only)",
                    r.pos - 1
                )));
            }
        }
    }

    // The top object is the OrderedDict of {name: tensor}.
    let top = stack
        .pop()
        .ok_or_else(|| Error::Msg("pickle: empty stack at STOP".into()))?;
    let Val::Dict(items) = top else {
        return Err(Error::Msg("pickle: top object is not a dict".into()));
    };
    let mut out = Vec::with_capacity(items.len());
    for (k, v) in items {
        let Val::Str(name) = k else {
            return Err(Error::Msg("pickle: non-string state_dict key".into()));
        };
        out.push((name, v));
    }
    Ok(out)
}

/// `('storage', <FloatStorage global>, <key str>, <location str>, <numel int>)` → `Val::Storage`.
fn resolve_persid(pid: Val) -> Result<Val> {
    let Val::Tuple(t) = pid else {
        return Err(Error::Msg("pickle: persid not a tuple".into()));
    };
    if t.len() < 3 {
        return Err(Error::Msg("pickle: persid tuple too short".into()));
    }
    let dtype = match &t[1] {
        Val::Global(g) => StorageDtype::from_global(g)?,
        _ => {
            return Err(Error::Msg(
                "pickle: persid storage type not a global".into(),
            ))
        }
    };
    let key = match &t[2] {
        Val::Str(s) => s.clone(),
        Val::Int(n) => n.to_string(),
        _ => return Err(Error::Msg("pickle: persid storage key not a string".into())),
    };
    Ok(Val::Storage { dtype, key })
}

/// Apply a `REDUCE`: only `OrderedDict()` and `_rebuild_tensor_v2(...)` are produced by `torch.save`.
fn apply_reduce(callable: Val, args: Val) -> Result<Val> {
    let Val::Global(g) = callable else {
        return Err(Error::Msg("pickle: REDUCE callable not a global".into()));
    };
    match g.as_str() {
        "collections OrderedDict" => Ok(Val::Dict(Vec::new())),
        "torch._utils _rebuild_tensor_v2" => {
            let Val::Tuple(a) = args else {
                return Err(Error::Msg(
                    "pickle: _rebuild_tensor_v2 args not a tuple".into(),
                ));
            };
            // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, ...])
            if a.len() < 4 {
                return Err(Error::Msg("pickle: _rebuild_tensor_v2 too few args".into()));
            }
            let (dtype, key) = match &a[0] {
                Val::Storage { dtype, key } => (*dtype, key.clone()),
                _ => {
                    return Err(Error::Msg(
                        "pickle: rebuild_tensor arg0 not a storage".into(),
                    ))
                }
            };
            let offset = match &a[1] {
                Val::Int(n) => *n,
                _ => return Err(Error::Msg("pickle: rebuild_tensor offset not int".into())),
            };
            let size = int_tuple(&a[2])?;
            let stride = int_tuple(&a[3])?;
            Ok(Val::Tensor {
                dtype,
                key,
                offset,
                size,
                stride,
            })
        }
        other => Err(Error::Msg(format!(
            "pickle: unsupported REDUCE callable `{other}`"
        ))),
    }
}

fn int_tuple(v: &Val) -> Result<Vec<i64>> {
    let Val::Tuple(t) = v else {
        return Err(Error::Msg("pickle: expected an int tuple".into()));
    };
    t.iter()
        .map(|e| match e {
            Val::Int(n) => Ok(*n),
            _ => Err(Error::Msg("pickle: non-int in shape/stride tuple".into())),
        })
        .collect()
}

/// Whether `stride` is the C-contiguous (row-major) stride for `size` — so the tensor occupies a
/// plain `[offset .. offset+numel]` slice of its (possibly shared) storage. Size-1 axes may carry any
/// stride (PyTorch's contiguity rule). Non-contiguous views (e.g. transposes) are not produced for
/// saved weights, so they are rejected rather than silently mis-read.
fn is_c_contiguous(size: &[i64], stride: &[i64]) -> bool {
    if size.len() != stride.len() {
        return false;
    }
    let mut expected: i64 = 1;
    for i in (0..size.len()).rev() {
        if size[i] == 1 {
            continue;
        }
        if stride[i] != expected {
            return false;
        }
        expected *= size[i];
    }
    true
}

/// Decode a raw little-endian storage blob (in `dtype`) of `numel` elements to an f32 vector,
/// mirroring `torch.load(...).float()`.
fn decode_to_f32(bytes: &[u8], dtype: StorageDtype, numel: usize) -> Result<Vec<f32>> {
    let need = numel * dtype.elem_size();
    if bytes.len() < need {
        return Err(Error::Msg(format!(
            "storage blob too small: have {} bytes, need {} ({numel} × {})",
            bytes.len(),
            need,
            dtype.elem_size()
        )));
    }
    let out = match dtype {
        StorageDtype::Float32 => bytes[..need]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StorageDtype::BFloat16 => bytes[..need]
            .chunks_exact(2)
            // bf16 occupies the high 16 bits of the f32 — widen by a 16-bit left shift.
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        StorageDtype::Float16 => bytes[..need]
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        StorageDtype::Int64 => bytes[..need]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
    };
    Ok(out)
}

/// IEEE-754 half → f32 (no `half` crate dep).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (h & 0x8000) as u32;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 16
        } else {
            // subnormal — normalize
            let mut e: i32 = -1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 16) | (((127 - 15 + 1 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 16) | 0x7f80_0000 | (mant << 13) // inf / nan
    } else {
        (sign << 16) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Validate the externally-supplied tensor metadata from a `.pth` and compute the contiguous
/// `[start..end)` byte range (and element count `numel`) into its storage blob. A malformed
/// checkpoint must yield an `Err`, never a panic or a wrapped/huge index:
///
/// - every dimension is non-negative and fits an MLX `i32` axis (so the later `d as i32` shape cast
///   cannot truncate or sign-flip),
/// - the storage `offset` is non-negative,
/// - `numel`, `offset · elsize`, `numel · elsize`, and `start + len` are all computed with checked
///   arithmetic (no `usize`/`i64` overflow).
fn tensor_byte_range(
    name: &str,
    offset: i64,
    size: &[i64],
    elsize: usize,
) -> Result<(usize, usize, usize)> {
    if offset < 0 {
        return Err(Error::Msg(format!(
            "{name}: negative storage offset {offset}"
        )));
    }
    let mut numel: usize = 1;
    for &d in size {
        if d < 0 || d > i32::MAX as i64 {
            return Err(Error::Msg(format!(
                "{name}: invalid tensor dimension {d} in shape {size:?} (must be 0..={})",
                i32::MAX
            )));
        }
        numel = numel.checked_mul(d as usize).ok_or_else(|| {
            Error::Msg(format!(
                "{name}: element count overflows usize for shape {size:?}"
            ))
        })?;
    }
    let start = (offset as usize).checked_mul(elsize).ok_or_else(|| {
        Error::Msg(format!(
            "{name}: storage byte offset overflows usize (offset {offset}, elsize {elsize})"
        ))
    })?;
    let byte_len = numel.checked_mul(elsize).ok_or_else(|| {
        Error::Msg(format!(
            "{name}: storage byte length overflows usize (numel {numel}, elsize {elsize})"
        ))
    })?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| Error::Msg(format!("{name}: storage slice end overflows usize")))?;
    Ok((start, end, numel))
}

/// Enforce the module's documented **STORED-only** support scope (F-017): the `zip` crate
/// transparently inflates DEFLATE (and other) entries, so a tiny compressed entry could expand to
/// many GB — a decompression-bomb DoS — and `load_pth_f32` only checks the blob length *after* the
/// full `read_to_end`. A STORED entry's reader yields exactly its on-disk bytes (no amplification),
/// so refusing every other method bounds each read by the archive size.
fn require_stored(method: zip::CompressionMethod, who: &str) -> Result<()> {
    if method != zip::CompressionMethod::Stored {
        return Err(Error::Msg(format!(
            "{who}: zip entry uses {method:?} compression, but only STORED (uncompressed) entries \
             are supported — a compressed entry could inflate a tiny archive to many GB \
             (decompression-bomb DoS); re-export the checkpoint without zip compression"
        )));
    }
    Ok(())
}

/// Load a torch `.pth` checkpoint, returning every tensor as an f32 MLX [`Array`] in PyTorch layout
/// (`torch.load(...).float()` semantics). Conv-weight transposes + key renames are the caller's job.
pub fn load_pth_f32(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    load_pth_f32_filtered(path, |_| true)
}

/// [`load_pth_f32`] keeping only the tensors whose name passes `keep`: the others' storage blobs
/// are never read (a multi-tower checkpoint loads one tower without materializing the rest).
pub fn load_pth_f32_filtered(
    path: impl AsRef<Path>,
    keep: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Msg(format!("open {}: {e}", path.display())))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| Error::Msg(format!("read zip {}: {e}", path.display())))?;

    // Locate `<prefix>data.pkl` and derive the storage-blob prefix.
    let names: Vec<String> = zip
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect::<zip::result::ZipResult<_>>()
        .map_err(|e| Error::Msg(format!("read zip names {}: {e}", path.display())))?;
    let pkl_name = names
        .iter()
        .find(|n| n.ends_with("data.pkl"))
        .ok_or_else(|| Error::Msg(format!("{}: no data.pkl in archive", path.display())))?
        .clone();
    let prefix = pkl_name.strip_suffix("data.pkl").unwrap().to_string();

    let mut pkl_bytes = Vec::new();
    {
        let mut pkl_entry = zip
            .by_name(&pkl_name)
            .map_err(|e| Error::Msg(format!("read {pkl_name}: {e}")))?;
        require_stored(pkl_entry.compression(), &pkl_name)?;
        // F-071: don't trust the zip header's compressed_size for the reservation — a crafted
        // archive can claim a multi-hundred-GB entry and abort the process on allocation before a
        // byte is read (bypassing the bomb guards). `try_reserve_exact` surfaces that as a typed
        // error instead; `read_to_end` still grows the Vec to the real size as bytes arrive.
        pkl_bytes
            .try_reserve_exact(pkl_entry.compressed_size() as usize)
            .map_err(|_| Error::Msg(format!("read {pkl_name}: entry too large to reserve")))?;
        pkl_entry.read_to_end(&mut pkl_bytes)?;
    }

    let specs: Vec<(String, Val)> = parse_pickle(&pkl_bytes)?
        .into_iter()
        .filter(|(name, _)| keep(name))
        .collect();

    // How many tensors still reference each storage `key`. Multiple tensors can be views into one
    // shared storage blob (the case the offset-fix `460d3e7` exists for); without memoization they
    // would each re-open, re-read, and re-decode the whole multi-MB blob (F-018). Cache the blob bytes
    // by key and drop each one once its last referencing tensor has been decoded, so peak memory still
    // holds only the blobs currently in flight.
    let mut remaining: HashMap<String, usize> = HashMap::new();
    for (_, spec) in &specs {
        if let Val::Tensor { key, .. } = spec {
            *remaining.entry(key.clone()).or_insert(0) += 1;
        }
    }
    let mut blob_cache: HashMap<String, Vec<u8>> = HashMap::new();

    let mut out = HashMap::with_capacity(specs.len());
    for (name, spec) in specs {
        let Val::Tensor {
            dtype,
            key,
            offset,
            size,
            stride,
        } = spec
        else {
            // Non-tensor entries (rare) are skipped — the reference keeps only `torch.Tensor`s.
            continue;
        };
        // Validate the externally-supplied metadata and compute the contiguous byte range BEFORE the
        // contiguity check — a passing range guarantees `numel ≤ i64::MAX`, so `is_c_contiguous`'s
        // running i64 product (a subset of the dims) cannot overflow either.
        let elsize = dtype.elem_size();
        let (start, end, numel) = tensor_byte_range(&name, offset, &size, elsize)?;
        // A tensor is a view into its (possibly shared) storage at `offset`. For a C-contiguous view
        // that view is the byte slice `[offset .. offset+numel]`; non-contiguous strides are rejected.
        if !is_c_contiguous(&size, &stride) {
            return Err(Error::Msg(format!(
                "{name}: non-contiguous storage stride {stride:?} for size {size:?} unsupported"
            )));
        }
        // Read the storage blob once and cache it by key; later views of the same storage reuse it.
        if !blob_cache.contains_key(&key) {
            let blob_name = format!("{prefix}data/{key}");
            let mut blob = Vec::new();
            let mut blob_entry = zip
                .by_name(&blob_name)
                .map_err(|e| Error::Msg(format!("read storage {blob_name} for {name}: {e}")))?;
            require_stored(blob_entry.compression(), &blob_name)?;
            // F-071: try_reserve_exact (see the pkl entry above) — a crafted header must not abort.
            blob.try_reserve_exact(blob_entry.compressed_size() as usize)
                .map_err(|_| Error::Msg(format!("read storage {blob_name}: entry too large")))?;
            blob_entry.read_to_end(&mut blob)?;
            blob_cache.insert(key.clone(), blob);
        }
        let blob = &blob_cache[&key];
        if end > blob.len() {
            return Err(Error::Msg(format!(
                "{name}: storage slice [{start}..{end}] exceeds blob len {} (offset {offset}, numel {numel})",
                blob.len()
            )));
        }
        let floats = decode_to_f32(&blob[start..end], dtype, numel)?;
        // Dims are validated non-negative and ≤ i32::MAX by `tensor_byte_range`, so this cast cannot
        // truncate or sign-flip.
        let shape: Vec<i32> = size.iter().map(|&d| d as i32).collect();
        out.insert(name, Array::from_slice(&floats, &shape));

        // Drop the cached blob once its last referencing tensor has been decoded (F-018).
        if let Some(refs) = remaining.get_mut(&key) {
            *refs -= 1;
            if *refs == 0 {
                blob_cache.remove(&key);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_stored_rejects_compressed_entries() {
        // F-017: only STORED entries are accepted; a DEFLATE entry (the decompression-bomb vector)
        // is rejected with a clear error before any read.
        assert!(require_stored(zip::CompressionMethod::Stored, "x").is_ok());
        // `DEFLATE` is `Unsupported(8)` here (the crate is built `default-features = false`, no
        // inflater) — still the canonical non-STORED case to reject.
        let err = require_stored(zip::CompressionMethod::DEFLATE, "data.pkl")
            .unwrap_err()
            .to_string();
        assert!(err.contains("STORED"), "got: {err}");
        assert!(err.contains("decompression-bomb"), "got: {err}");
    }

    #[test]
    fn f16_round_trip_known_values() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -0.5 = 0xB800, 0 = 0x0000
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xB800), -0.5);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }

    #[test]
    fn bf16_decode_high_bits() {
        // bf16 1.0 = 0x3F80 → f32 1.0; -2.0 bf16 = 0xC000 → f32 -2.0
        let v = decode_to_f32(&[0x80, 0x3F, 0x00, 0xC0], StorageDtype::BFloat16, 2).unwrap();
        assert_eq!(v, vec![1.0, -2.0]);
    }

    #[test]
    fn f32_decode_little_endian() {
        let v = decode_to_f32(&1.5f32.to_le_bytes(), StorageDtype::Float32, 1).unwrap();
        assert_eq!(v, vec![1.5]);
    }

    #[test]
    fn tensor_byte_range_valid() {
        // offset 2 elements, shape [3,4] (numel 12), f32 (elsize 4): start 8, len 48, end 56.
        assert_eq!(tensor_byte_range("t", 2, &[3, 4], 4).unwrap(), (8, 56, 12));
        // Scalar (empty shape) is numel 1.
        assert_eq!(tensor_byte_range("t", 0, &[], 4).unwrap(), (0, 4, 1));
        // A zero dimension collapses numel to 0 (an empty tensor), not an error.
        assert_eq!(tensor_byte_range("t", 0, &[0, 5], 4).unwrap(), (0, 0, 0));
    }

    #[test]
    fn tensor_byte_range_rejects_negative_offset() {
        let err = tensor_byte_range("t", -1, &[2], 4).unwrap_err().to_string();
        assert!(err.contains("negative storage offset"), "{err}");
    }

    #[test]
    fn tensor_byte_range_rejects_negative_or_oversized_dims() {
        // A negative dimension (would otherwise sign-flip the i32 shape cast).
        let neg = tensor_byte_range("t", 0, &[-1, 2], 4)
            .unwrap_err()
            .to_string();
        assert!(neg.contains("invalid tensor dimension"), "{neg}");
        // A dimension that does not fit an MLX i32 axis (would otherwise truncate).
        let big = tensor_byte_range("t", 0, &[i32::MAX as i64 + 1], 4)
            .unwrap_err()
            .to_string();
        assert!(big.contains("invalid tensor dimension"), "{big}");
    }

    #[test]
    fn tensor_byte_range_rejects_overflowing_products() {
        // numel = i32::MAX³ overflows usize (the unchecked `.product()` would have wrapped).
        let m = i32::MAX as i64;
        let numel = tensor_byte_range("t", 0, &[m, m, m], 4)
            .unwrap_err()
            .to_string();
        assert!(numel.contains("element count overflows usize"), "{numel}");
        // offset · elsize overflows usize (a huge — but in-range non-negative — offset).
        let off = tensor_byte_range("t", i64::MAX, &[1], 4)
            .unwrap_err()
            .to_string();
        assert!(off.contains("storage byte offset overflows usize"), "{off}");
    }

    /// A hand-built minimal protocol-2 pickle of `OrderedDict({"a": rebuild_tensor(FloatStorage 0,
    /// offset 0, size (2,), stride (1,), False, {})})` exercises the VM end to end.
    #[test]
    fn parse_minimal_state_dict() {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&[0x80, 2]); // PROTO 2
                                         // OrderedDict()
        p.push(b'c');
        p.extend_from_slice(b"collections\nOrderedDict\n");
        p.push(b')'); // EMPTY_TUPLE
        p.push(b'R'); // REDUCE → dict
        p.push(b'('); // MARK
        p.push(b'X'); // BINUNICODE "a"
        p.extend_from_slice(&1u32.to_le_bytes());
        p.push(b'a');
        // _rebuild_tensor_v2
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(b'('); // MARK (args)
                      // storage tuple ('storage', FloatStorage, '0', 'cpu', 2)
        p.push(b'('); // MARK
        p.push(b'X');
        p.extend_from_slice(&7u32.to_le_bytes());
        p.extend_from_slice(b"storage");
        p.push(b'c');
        p.extend_from_slice(b"torch\nFloatStorage\n");
        p.push(b'X');
        p.extend_from_slice(&1u32.to_le_bytes());
        p.push(b'0');
        p.push(b'X');
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(b"cpu");
        p.push(b'K');
        p.push(2); // numel
        p.push(b't'); // TUPLE → storage tuple
        p.push(b'Q'); // BINPERSID
        p.push(b'K');
        p.push(0); // offset
        p.push(b'K');
        p.push(2);
        p.push(0x85); // TUPLE1 → size (2,)
        p.push(b'K');
        p.push(1);
        p.push(0x85); // TUPLE1 → stride (1,)
        p.push(0x89); // NEWFALSE requires_grad
        p.push(b'}'); // EMPTY_DICT backward_hooks
        p.push(b't'); // TUPLE → args
        p.push(b'R'); // REDUCE → tensor
        p.push(b'u'); // SETITEMS (pops back to the MARK after the dict)
        p.push(b'.'); // STOP

        let specs = parse_pickle(&p).unwrap();
        assert_eq!(specs.len(), 1);
        let (name, v) = &specs[0];
        assert_eq!(name, "a");
        match v {
            Val::Tensor {
                dtype,
                key,
                offset,
                size,
                stride,
            } => {
                assert_eq!(*dtype, StorageDtype::Float32);
                assert_eq!(key, "0");
                assert_eq!(*offset, 0);
                assert_eq!(size, &vec![2]);
                assert_eq!(stride, &vec![1]);
            }
            _ => panic!("expected a tensor spec"),
        }
    }

    /// F-016: a LONG1 length > 8 would shift by >= 64 (`(b as i64) << (8*i)`), panicking in debug and
    /// silently masking to a wrong value in release. It must be rejected with a typed error instead.
    #[test]
    fn parse_rejects_oversized_long1() {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&[0x80, 2]); // PROTO 2
        p.push(0x8a); // LONG1
        p.push(9); // length 9 > 8
        p.extend_from_slice(&[0u8; 9]);
        p.push(b'.'); // STOP (unreached — the length check fires first)

        let err = parse_pickle(&p).unwrap_err().to_string();
        assert!(
            err.contains("LONG1 length 9 exceeds 8 bytes"),
            "expected a LONG1-length error, got: {err}"
        );
    }

    #[test]
    fn count_nodes_counts_nested_structure() {
        assert_eq!(count_nodes(&Val::Int(7)), 1);
        // Tuple(Tuple(Int, Int), Int) → 1 (outer) + [1 (inner) + 1 + 1] + 1 = 5
        let nested = Val::Tuple(vec![
            Val::Tuple(vec![Val::Int(1), Val::Int(2)]),
            Val::Int(3),
        ]);
        assert_eq!(count_nodes(&nested), 5);
    }

    /// F-015: a crafted pickle that `BINGET`s a memoized value twice, pairs the copies with `TUPLE2`,
    /// and `BINPUT`s the result back doubles the in-memory structure every iteration. Without a budget
    /// this demands `2^n` memory; the node-clone budget must make it error (well before OOM) instead.
    #[test]
    fn parse_rejects_billion_laughs_amplification() {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&[0x80, 2]); // PROTO 2
        p.push(b')'); // EMPTY_TUPLE — the seed value
        p.push(b'q');
        p.push(0); // BINPUT 0  → memo[0] = ()
                   // 40 doublings is far past the 8M-node budget; the parse must bail mid-way.
        for _ in 0..40 {
            p.push(b'h');
            p.push(0); // BINGET 0
            p.push(b'h');
            p.push(0); // BINGET 0
            p.push(0x86); // TUPLE2 → (memo0, memo0)
            p.push(b'q');
            p.push(0); // BINPUT 0  → memo[0] doubles
        }
        p.push(b'.'); // STOP

        let err = parse_pickle(&p).unwrap_err().to_string();
        assert!(
            err.contains("node budget exceeded"),
            "expected a node-budget error, got: {err}"
        );
    }

    #[test]
    fn c_contiguous_check() {
        // standard row-major strides
        assert!(is_c_contiguous(&[160, 12, 3, 3, 3], &[324, 27, 9, 3, 1]));
        assert!(is_c_contiguous(&[4, 5], &[5, 1]));
        // size-1 axes may carry any stride
        assert!(is_c_contiguous(&[3, 1, 1, 1], &[1, 3, 3, 3]));
        // a transposed (non-contiguous) view is rejected
        assert!(!is_c_contiguous(&[4, 5], &[1, 4]));
        // scalar
        assert!(is_c_contiguous(&[], &[]));
    }

    /// sc-24833: a real `torch.save`d module state dict with a BatchNorm (float tensors, the int64
    /// `num_batches_tracked` counter, the `_metadata` attribute) — the format of the published
    /// E-LatentLPIPS checkpoints — loads with every float tensor bit-identical to the producer's
    /// formula values and the counter decoded. Mutation: drop the `LongStorage` arm ⇒ the load
    /// errors ⇒ red.
    #[test]
    fn loads_a_torch_state_dict_with_batchnorm_counters() {
        use crate::train::formula::{formula_values, Role};
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../contracts/gen-core/tests/fixtures/latent_perceptual/tiny_bn_state_dict.pth",
        );
        let sd = load_pth_f32(path).unwrap();
        assert_eq!(sd.len(), 7, "{:?}", sd.keys().collect::<Vec<_>>());
        for (k, shape, role) in [
            ("0.weight", vec![2, 4, 3, 3], Role::Conv),
            ("0.bias", vec![2], Role::Bias),
            ("1.weight", vec![2], Role::NormWeight),
            ("1.bias", vec![2], Role::Bias),
            ("1.running_mean", vec![2], Role::BnMean),
            ("1.running_var", vec![2], Role::BnVar),
        ] {
            let a = &sd[k];
            assert_eq!(a.shape(), shape.as_slice(), "{k}");
            assert_eq!(
                a.as_slice::<f32>(),
                formula_values(k, &shape, role).as_slice(),
                "{k}"
            );
        }
        assert_eq!(sd["1.num_batches_tracked"].item::<f32>(), 7.0);
    }
}
