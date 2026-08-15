fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = sceneworks_core::jobs_store::backend_capability_matrix()?;
    let output = format!("{}\n", serde_json::to_string_pretty(&matrix)?);
    if let Some(path) = std::env::args_os().nth(1) {
        std::fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}
