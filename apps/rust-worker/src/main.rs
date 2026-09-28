fn main() -> Result<(), sceneworks_worker::WorkerError> {
    sceneworks_worker::run_on_worker_entry_thread(sceneworks_worker::run)
}
