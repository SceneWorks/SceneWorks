//! Post-pin SC-22261 hostile-corpus runner for the production SVG sanitizer.
//! The binary delegates every decision to `vector_jobs`; it adds no sanitizer.

use std::{
    env,
    path::{Path, PathBuf},
};

use sceneworks_worker::{terminal_sanitize_svg_bytes, terminal_write_sanitized_pair};
use serde_json::json;
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

async fn sanitize_file(input: &Path, output: &Path) -> Result<serde_json::Value, String> {
    let bytes = tokio::fs::read(input)
        .await
        .map_err(|error| format!("read {}: {error}", input.display()))?;
    match terminal_sanitize_svg_bytes(&bytes) {
        Err(error) => Ok(
            json!({"outcome":"rejected","error_code":error,"canonical_svg_sha256":null,"preview_png_sha256":null,"published_paths":[],"staging_residue":[],"result_contains_inline_svg":false}),
        ),
        Ok(value) => match terminal_write_sanitized_pair(&value, output).await {
            Err(error) => Err(format!("publish {}: {error}", output.display())),
            Ok((svg, preview)) => {
                let svg_bytes = tokio::fs::read(&svg)
                    .await
                    .map_err(|error| format!("read {}: {error}", svg.display()))?;
                let preview_bytes = tokio::fs::read(&preview)
                    .await
                    .map_err(|error| format!("read {}: {error}", preview.display()))?;
                Ok(
                    json!({"outcome":"sanitized_inert","error_code":"sanitized_inert","canonical_svg_sha256":hash(&svg_bytes),"preview_png_sha256":hash(&preview_bytes),"published_paths":["canonical.svg","preview.png"],"staging_residue":[],"result_contains_inline_svg":false}),
                )
            }
        },
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 || args[1] != "run" {
        eprintln!("usage: starvector-terminal-sanitize run <input-svg> <output-directory>");
        std::process::exit(2);
    }
    let input = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    match sanitize_file(&input, &output).await {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("starvector terminal sanitizer infrastructure error: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_file;

    const INERT_SVG: &[u8] = b"<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#00ff00\"/></svg>";

    #[tokio::test]
    async fn sanitizer_refusal_is_a_structured_policy_rejection() {
        let temp = tempfile::tempdir().expect("temp dir");
        let input = temp.path().join("invalid.svg");
        tokio::fs::write(&input, [0xff]).await.expect("write input");
        let output = temp.path().join("published");
        let result = sanitize_file(&input, &output).await.expect("policy result");
        assert_eq!(result["outcome"], "rejected");
        assert!(!output.exists(), "rejected input must publish nothing");
    }

    #[tokio::test]
    async fn input_and_publication_failures_are_not_policy_rejections() {
        let temp = tempfile::tempdir().expect("temp dir");
        let missing = temp.path().join("missing.svg");
        assert!(sanitize_file(&missing, &temp.path().join("published"))
            .await
            .is_err());

        let input = temp.path().join("valid.svg");
        tokio::fs::write(&input, INERT_SVG)
            .await
            .expect("write input");
        let blocked_parent = temp.path().join("blocked-parent");
        tokio::fs::write(&blocked_parent, b"not a directory")
            .await
            .expect("write blocker");
        assert!(sanitize_file(&input, &blocked_parent.join("published"))
            .await
            .is_err());
    }
}
