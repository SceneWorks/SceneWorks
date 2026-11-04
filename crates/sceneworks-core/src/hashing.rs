//! Shared bounded SHA-256, lowercase hex and OS-random identifiers.
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

/// Lowercase hexadecimal encoding, including leading zeroes.
pub fn hex(bytes: &[u8]) -> String {
    hex_prefix(bytes, bytes.len() * 2)
}

/// Encode at most `chars` hexadecimal digits. A zero limit preserves the historical
/// prefix callers' convention of encoding the entire input.
pub fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(chars);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        if result.len() == chars {
            break;
        }
        result.push(DIGITS[(byte & 15) as usize] as char);
        if result.len() == chars {
            break;
        }
    }
    result
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Events around each bounded read, including the final empty read.
pub enum ReadEvent<'a> {
    Before,
    After(&'a [u8]),
}

/// Hash a stream while allowing callers to check cancellation or inspect each chunk.
fn update_sha256_with<E>(
    mut reader: impl Read,
    digest: &mut Sha256,
    mut observe: impl FnMut(u64, ReadEvent<'_>) -> Result<(), E>,
    mut read_error: impl FnMut(io::Error) -> E,
) -> Result<u64, E> {
    let mut count = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        observe(count, ReadEvent::Before)?;
        let read = reader.read(&mut buffer).map_err(&mut read_error)?;
        observe(count, ReadEvent::After(&buffer[..read]))?;
        if read == 0 {
            break;
        }
        count = count.checked_add(read as u64).ok_or_else(|| {
            read_error(io::Error::new(
                io::ErrorKind::InvalidData,
                "stream size overflows u64",
            ))
        })?;
        digest.update(&buffer[..read]);
    }
    Ok(count)
}

/// Append a stream to an existing digest without finalizing it (for composite protocols).
pub fn update_sha256(reader: impl Read, digest: &mut Sha256) -> io::Result<()> {
    update_sha256_with(reader, digest, |_, _| Ok(()), |error| error).map(|_| ())
}

/// Hash a stream while allowing callers to check cancellation or inspect each chunk.
pub fn sha256_reader_with(
    reader: impl Read,
    observe: impl FnMut(u64, ReadEvent<'_>) -> io::Result<()>,
) -> io::Result<(String, u64)> {
    try_sha256_reader_with(reader, observe, |error| error)
}

/// As `sha256_reader_with`, with caller-specific read and observation errors.
pub fn try_sha256_reader_with<E>(
    reader: impl Read,
    observe: impl FnMut(u64, ReadEvent<'_>) -> Result<(), E>,
    read_error: impl FnMut(io::Error) -> E,
) -> Result<(String, u64), E> {
    let mut digest = Sha256::new();
    let count = update_sha256_with(reader, &mut digest, observe, read_error)?;
    Ok((hex(&digest.finalize()), count))
}

/// Hash a reader from its current position using bounded memory; return bytes read as well.
pub fn sha256_reader(reader: impl Read) -> io::Result<(String, u64)> {
    sha256_reader_with(reader, |_, _| Ok(()))
}

/// Hash as `std::io::copy` does, retrying interrupted reads.
pub fn sha256_reader_retry(reader: impl Read) -> io::Result<(String, u64)> {
    struct Retry<R>(R);
    impl<R: Read> Read for Retry<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            loop {
                match self.0.read(buffer) {
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    result => return result,
                }
            }
        }
    }
    sha256_reader(Retry(reader))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    sha256_reader(File::open(path)?).map(|(digest, _)| digest)
}

/// Return OS-random bytes encoded as lowercase hex; propagate RNG failure.
pub fn random_hex(bytes: usize) -> Result<String, getrandom::Error> {
    let mut buffer = vec![0; bytes];
    getrandom::fill(&mut buffer)?;
    Ok(hex(&buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vectors_and_streams() {
        assert_eq!(hex(&[0, 1, 15, 16, 255]), "00010f10ff");
        assert_eq!(hex_prefix(&[0xab, 0xcd], 3), "abc");
        assert_eq!(hex_prefix(&[0xab, 0xcd], 0), "abcd");
        assert_eq!(hex_prefix(&[0xab], 8), "ab");
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_reader(&b"abc"[..]).unwrap(),
            (sha256_bytes(b"abc"), 3)
        );
        assert_eq!(
            sha256_reader((&b"abcdef"[..]).take(3)).unwrap().0,
            sha256_bytes(b"abc")
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(b"abc"));
        assert_eq!(
            sha256_file(&dir.path().join("missing")).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        let mut composite = Sha256::new();
        composite.update(b"prefix");
        update_sha256(&b"abc"[..], &mut composite).unwrap();
        assert_eq!(hex(&composite.finalize()), sha256_bytes(b"prefixabc"));
        let canceled = sha256_reader_with(&b"abc"[..], |_, event| match event {
            ReadEvent::After(bytes) if !bytes.is_empty() => {
                Err(io::Error::new(io::ErrorKind::Interrupted, "canceled"))
            }
            _ => Ok(()),
        });
        assert_eq!(canceled.unwrap_err().kind(), io::ErrorKind::Interrupted);
        let large = vec![0x5a; 2 * 64 * 1024 + 19];
        assert_eq!(
            sha256_reader(&large[..]).unwrap(),
            (sha256_bytes(&large), large.len() as u64)
        );
        struct InterruptedOnce(u8);
        impl Read for InterruptedOnce {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.0 += 1;
                match self.0 {
                    1 => Err(io::ErrorKind::Interrupted.into()),
                    2 => {
                        bytes[0] = b'x';
                        Ok(1)
                    }
                    _ => Ok(0),
                }
            }
        }
        assert_eq!(
            sha256_reader_retry(InterruptedOnce(0)).unwrap().0,
            sha256_bytes(b"x")
        );
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("broken"))
            }
        }
        assert_eq!(sha256_reader(Broken).unwrap_err().to_string(), "broken");
        let value = random_hex(16).unwrap();
        assert_eq!(value.len(), 32);
        assert!(value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
    }
}
