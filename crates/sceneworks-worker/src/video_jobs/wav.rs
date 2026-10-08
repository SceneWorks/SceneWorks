//! The PCM16 WAV writer every audio output uses (LTX synchronized audio, the audio lane, YuE2),
//! split out of the shared `video_jobs` parent. `write_wav_pcm16_with_info` adds a RIFF
//! `LIST`/`INFO` chunk so a licence note can travel inside the file (sc-23000).

use std::path::Path;

use super::AudioTrack;
use crate::WorkerResult;

/// Write f32 PCM to a canonical 16-bit WAV. Signals already within `[-1, 1]` retain their original
/// amplitude; only over-range input is peak-normalized to prevent clipping. `pub(crate)` so the
/// pure-audio job path reuses it (sc-13404).
pub(crate) fn write_wav_pcm16(audio: &AudioTrack, path: &Path) -> WorkerResult<()> {
    write_wav_pcm16_with_info(audio, path, &[])
}

/// [`write_wav_pcm16`] plus a RIFF `LIST`/`INFO` chunk carrying `(four-cc, text)` tags (e.g.
/// `ICOP` copyright, `ICMT` comment) — so a licence note travels inside the file itself (sc-23000).
/// The chunk follows `data`, so every chunk-walking reader in the tree (which stops at `data`) and
/// any player reads the audio unchanged. An empty tag list writes exactly the canonical file.
pub(crate) fn write_wav_pcm16_with_info(
    audio: &AudioTrack,
    path: &Path,
    info: &[([u8; 4], &str)],
) -> WorkerResult<()> {
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    let scale = i16::MAX as f32 / peak.max(1.0);
    let mut pcm = Vec::with_capacity(audio.samples.len() * 2);
    for &sample in &audio.samples {
        let value = (sample * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        pcm.extend_from_slice(&value.to_le_bytes());
    }

    let channels = audio.channels.max(1);
    let bits_per_sample = 16u16;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = audio.sample_rate * block_align as u32;
    let data_len = pcm.len() as u32;

    let list = riff_info_list(info);
    let mut buffer = Vec::with_capacity(44 + pcm.len() + list.len());
    buffer.extend_from_slice(b"RIFF");
    // The data body is PCM16, so it is always even-sized: no pad byte before the LIST chunk.
    buffer.extend_from_slice(&(36 + data_len + list.len() as u32).to_le_bytes());
    buffer.extend_from_slice(b"WAVE");
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buffer.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buffer.extend_from_slice(&channels.to_le_bytes());
    buffer.extend_from_slice(&audio.sample_rate.to_le_bytes());
    buffer.extend_from_slice(&byte_rate.to_le_bytes());
    buffer.extend_from_slice(&block_align.to_le_bytes());
    buffer.extend_from_slice(&bits_per_sample.to_le_bytes());
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_len.to_le_bytes());
    buffer.extend_from_slice(&pcm);
    buffer.extend_from_slice(&list);
    std::fs::write(path, buffer)?;
    Ok(())
}

/// A `LIST`/`INFO` chunk for `info` (empty when there are no tags). Each text is NUL-terminated
/// and word-aligned, as RIFF requires.
fn riff_info_list(info: &[([u8; 4], &str)]) -> Vec<u8> {
    if info.is_empty() {
        return Vec::new();
    }
    let mut body = b"INFO".to_vec();
    for (id, text) in info {
        let mut value = text.as_bytes().to_vec();
        value.push(0);
        body.extend_from_slice(id);
        body.extend_from_slice(&(value.len() as u32).to_le_bytes());
        let odd = value.len() & 1 == 1;
        body.extend_from_slice(&value);
        if odd {
            body.push(0);
        }
    }
    let mut list = b"LIST".to_vec();
    list.extend_from_slice(&(body.len() as u32).to_le_bytes());
    list.extend_from_slice(&body);
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tagged WAV keeps a valid RIFF size and decodes exactly like the canonical one; odd-length
    /// tag text is word-aligned. Mutation that reds this: leaving the LIST length out of the RIFF
    /// size.
    #[test]
    fn info_tags_follow_data_and_keep_the_file_valid() {
        let dir = tempfile::tempdir().unwrap();
        let track = AudioTrack {
            samples: vec![0.25, -0.25, 0.5, -0.5],
            sample_rate: 8_000,
            channels: 2,
        };
        let plain_path = dir.path().join("plain.wav");
        let tagged_path = dir.path().join("tagged.wav");
        write_wav_pcm16(&track, &plain_path).unwrap();
        write_wav_pcm16_with_info(
            &track,
            &tagged_path,
            &[(*b"ICOP", "odd"), (*b"ICMT", "even")],
        )
        .unwrap();
        let plain = std::fs::read(&plain_path).unwrap();
        let bytes = std::fs::read(&tagged_path).unwrap();
        assert_eq!(plain.len(), 44 + 8, "no tags: the canonical file");
        let riff = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff + 8, bytes.len(), "the RIFF size covers the LIST chunk");
        assert_eq!(&bytes[44..52], &plain[44..], "the audio is unchanged");
        assert_eq!(&bytes[52..56], b"LIST");
        assert_eq!(bytes.len() % 2, 0, "word-aligned");
        let decoded = crate::audio_jobs::read_wav_pcm16(&tagged_path).unwrap();
        assert_eq!(decoded.samples.len(), 4);
    }
}
