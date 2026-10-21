fn main() -> Result<(), sceneworks_worker::WorkerError> {
    // A multi-rank Iris training job (sc-25689) re-enters this executable once per rank; such a
    // process trains its rank and exits before any worker machinery starts.
    if let Some(code) = sceneworks_worker::iris_rank_entry() {
        std::process::exit(code);
    }
    sceneworks_worker::run_on_worker_entry_thread(sceneworks_worker::run)
}
