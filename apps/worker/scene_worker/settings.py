import os
from pathlib import Path


class WorkerSettings:
    def __init__(self) -> None:
        self.worker_id = os.getenv("SCENEWORKS_WORKER_ID", "worker-local-0")
        self.gpu_id = os.getenv("SCENEWORKS_GPU_ID", "auto")
        self.api_url = os.getenv("SCENEWORKS_API_URL", "http://localhost:8000")
        self.data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).resolve()
        self.config_dir = Path(os.getenv("SCENEWORKS_CONFIG_DIR", "config")).resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        self.heartbeat_seconds = int(os.getenv("SCENEWORKS_WORKER_HEARTBEAT_SECONDS", "30"))
        self.poll_seconds = int(os.getenv("SCENEWORKS_WORKER_POLL_SECONDS", "3"))

    def for_worker(self, *, worker_id: str, gpu_id: str) -> "WorkerSettings":
        settings = object.__new__(WorkerSettings)
        settings.worker_id = worker_id
        settings.gpu_id = gpu_id
        settings.api_url = self.api_url
        settings.data_dir = self.data_dir
        settings.config_dir = self.config_dir
        settings.access_token = self.access_token
        settings.heartbeat_seconds = self.heartbeat_seconds
        settings.poll_seconds = self.poll_seconds
        return settings
