//! Shared multimodal-LLM tensor helpers.
//!
//! Generic, model-agnostic operations for vision-language prompt encoders: splicing projected image
//! features into a language model's token embeddings at the image-token positions. The per-model VLM
//! stack (vision tower, projector, chat format, decode) lives in the LLM engine
//! ([`mlx-llm`](https://github.com/SceneWorks/mlx-llm)); this helper stays in `mlx-gen` because
//! several diffusion crates reuse it for their own multimodal prompt encoders (e.g. FLUX.2 caption
//! upsampling). It is a plain tensor gather — not specific to any one model (sc-7265, relocated from
//! the retired in-core JoyCaption module).

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::array::host_i32;
use crate::{Error, Result};

/// Splice projected image features into token embeddings. `input_ids` must already be expanded so
/// the number of image-token positions equals the number of projected image rows.
///
/// `token_embeds` is `[batch, seq, hidden]`; `projected_features` is `[batch, image_seq, hidden]` or
/// `[image_seq, hidden]`. Each row whose id equals `image_token_id` is replaced by the next
/// projected image row (appended after the text rows); every other row passes through unchanged.
pub fn splice_image_features(
    token_embeds: &Array,
    input_ids: &Array,
    projected_features: &Array,
    image_token_id: i32,
) -> Result<Array> {
    let sh = token_embeds.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "image splice: token embeddings must be [batch, seq, hidden], got {sh:?}"
        )));
    }
    let (b, s, h) = (sh[0], sh[1], sh[2]);
    let n_text = b * s;
    let fsh = projected_features.shape();
    let features = match fsh {
        [fb, fs, fh] if *fb == b && *fh == h => projected_features.reshape(&[fb * fs, h])?,
        [fs, fh] if *fh == h => projected_features.reshape(&[*fs, h])?,
        _ => {
            return Err(Error::Msg(format!(
                "image splice: projected features must be [batch, image_seq, {h}] or [image_seq, {h}], got {fsh:?}"
            )));
        }
    };
    let n_vis = features.shape()[0];
    let ids = host_i32(input_ids)?;
    let gather = image_gather_index(&ids, image_token_id, n_vis, n_text)?;
    let embeds_flat = token_embeds.reshape(&[n_text, h])?;
    let src = concatenate_axis(&[&embeds_flat, &features], 0)?;
    let idx = Array::from_slice(&gather, &[n_text]);
    Ok(src.take_axis(&idx, 0)?.reshape(&[b, s, h])?)
}

/// Build the gather index that maps each output row to its source: a text position `p` keeps row
/// `p`; the `k`-th image-token position maps to the appended row `n_text + k`. Validates that the id
/// length matches the embedding rows and the image-token count matches the projected image rows.
pub fn image_gather_index(
    ids: &[i32],
    image_token_id: i32,
    n_vis: i32,
    n_text: i32,
) -> Result<Vec<i32>> {
    if ids.len() != n_text as usize {
        return Err(Error::Msg(format!(
            "image splice: input_ids length {} does not match embedding rows {n_text}",
            ids.len()
        )));
    }
    let count = ids.iter().filter(|&&id| id == image_token_id).count() as i32;
    if count != n_vis {
        return Err(Error::Msg(format!(
            "image splice: image token count {count} does not match projected image rows {n_vis}"
        )));
    }
    let mut out = Vec::with_capacity(n_text as usize);
    let mut vi = 0i32;
    for (p, &id) in ids.iter().enumerate() {
        if id == image_token_id {
            out.push(n_text + vi);
            vi += 1;
        } else {
            out.push(p as i32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_index_maps_image_rows_to_appended_features() {
        // ids [10, IMG, IMG, 11]: 4 text rows, 2 image rows appended at indices 4 and 5.
        let got = image_gather_index(&[10, 77, 77, 11], 77, 2, 4).unwrap();
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_index_rejects_image_token_count_mismatch() {
        // One image token in the ids, but two projected image rows claimed.
        assert!(image_gather_index(&[77, 7], 77, 2, 2).is_err());
    }

    #[test]
    fn gather_index_rejects_id_length_mismatch() {
        // 3 ids but only 2 embedding rows declared.
        assert!(image_gather_index(&[77, 7, 8], 77, 1, 2).is_err());
    }
}
