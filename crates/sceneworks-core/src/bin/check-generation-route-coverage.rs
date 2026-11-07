//! Metadata only: never constructs an engine or loads weights.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = sceneworks_core::jobs_store::generation_route_coverage()?;
    let output = format!("{}\n", serde_json::to_string_pretty(&report)?);
    if let Some(path) = std::env::args_os().nth(1) {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    if !report.missing.is_empty() {
        return Err(format!(
            "{} legacy route cells lost their binding",
            report.missing.len()
        )
        .into());
    }
    Ok(())
}
