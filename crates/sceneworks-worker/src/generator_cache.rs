use std::path::PathBuf;
use std::sync::{mpsc, OnceLock};
use std::thread;

use mlx_gen::{
    AdapterKind, AdapterSpec, Generator, LoadSpec, MoeExpert, Precision, Quant, WeightsSource,
};
use tokio::sync::oneshot;

use crate::{WorkerError, WorkerResult};

type GeneratorJob = Box<dyn FnOnce(&mut GeneratorCache) + Send + 'static>;

static GENERATOR_WORKER: OnceLock<mpsc::Sender<GeneratorJob>> = OnceLock::new();

struct GeneratorCache {
    entry: Option<GeneratorCacheEntry>,
}

struct GeneratorCacheEntry {
    key: GeneratorCacheKey,
    generator: Box<dyn Generator>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GeneratorCacheKey {
    engine_id: String,
    weights: CacheWeightsSource,
    quantize: Option<Quant>,
    precision: Precision,
    control: Option<CacheWeightsSource>,
    extra_controls: Vec<CacheWeightsSource>,
    ip_adapter: Option<CacheWeightsSource>,
    adapters: Vec<CacheAdapterSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CacheWeightsSource {
    Dir(PathBuf),
    File(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheAdapterSpec {
    path: PathBuf,
    scale_bits: u32,
    kind: AdapterKind,
    pass_scale_bits: Option<Vec<u32>>,
    moe_expert: Option<MoeExpert>,
}

impl GeneratorCacheKey {
    pub(crate) fn from_load_spec(engine_id: &str, spec: &LoadSpec) -> Self {
        Self {
            engine_id: engine_id.to_owned(),
            weights: CacheWeightsSource::from(&spec.weights),
            quantize: spec.quantize,
            precision: spec.precision,
            control: spec.control.as_ref().map(CacheWeightsSource::from),
            extra_controls: spec
                .extra_controls
                .iter()
                .map(CacheWeightsSource::from)
                .collect(),
            ip_adapter: spec.ip_adapter.as_ref().map(CacheWeightsSource::from),
            adapters: spec.adapters.iter().map(CacheAdapterSpec::from).collect(),
        }
    }
}

impl From<&WeightsSource> for CacheWeightsSource {
    fn from(source: &WeightsSource) -> Self {
        match source {
            WeightsSource::Dir(path) => Self::Dir(path.clone()),
            WeightsSource::File(path) => Self::File(path.clone()),
        }
    }
}

impl From<&AdapterSpec> for CacheAdapterSpec {
    fn from(spec: &AdapterSpec) -> Self {
        Self {
            path: spec.path.clone(),
            scale_bits: spec.scale.to_bits(),
            kind: spec.kind,
            pass_scale_bits: spec
                .pass_scales
                .as_ref()
                .map(|scales| scales.iter().map(|scale| scale.to_bits()).collect()),
            moe_expert: spec.moe_expert,
        }
    }
}

impl GeneratorCache {
    fn new() -> Self {
        Self { entry: None }
    }

    fn with_generator<R>(
        &mut self,
        key: GeneratorCacheKey,
        spec: LoadSpec,
        load_error_context: String,
        run: impl FnOnce(&dyn Generator) -> WorkerResult<R>,
    ) -> WorkerResult<R> {
        if self.entry.as_ref().map_or(true, |entry| entry.key != key) {
            self.entry = None;
            let generator = mlx_gen::load(&key.engine_id, &spec)
                .map_err(|error| WorkerError::Engine(format!("{load_error_context}: {error}")))?;
            self.entry = Some(GeneratorCacheEntry {
                key: key.clone(),
                generator,
            });
        }

        let generator = self
            .entry
            .as_ref()
            .expect("cache entry populated")
            .generator
            .as_ref();
        run(generator)
    }
}

fn generator_worker() -> &'static mpsc::Sender<GeneratorJob> {
    GENERATOR_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        thread::Builder::new()
            .name("sceneworks-mlx-generator-cache".to_owned())
            .spawn(move || {
                let mut cache = GeneratorCache::new();
                while let Ok(job) = rx.recv() {
                    job(&mut cache);
                }
            })
            .expect("start MLX generator cache worker");
        tx
    })
}

pub(crate) async fn with_cached_generator<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let key = GeneratorCacheKey::from_load_spec(engine_id, &spec);
    let load_error_context = load_error_context.into();
    let (reply_tx, reply_rx) = oneshot::channel::<WorkerResult<R>>();
    let job = Box::new(move |cache: &mut GeneratorCache| {
        let result = cache.with_generator(key, spec, load_error_context, run);
        let _ = reply_tx.send(result);
    });
    generator_worker()
        .send(job)
        .map_err(|_| WorkerError::Engine("MLX generator cache worker stopped".to_owned()))?;
    reply_rx.await.map_err(|_| {
        WorkerError::Engine("MLX generator cache worker dropped the job result".to_owned())
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_includes_adapter_fingerprint() {
        let base = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        let mut with_adapter = base.clone();
        with_adapter.adapters = vec![AdapterSpec::new(
            PathBuf::from("/loras/style.safetensors"),
            0.8,
            AdapterKind::Lora,
        )];
        let mut different_scale = with_adapter.clone();
        different_scale.adapters[0].scale = 0.9;

        assert_ne!(
            GeneratorCacheKey::from_load_spec("z_image_turbo", &base),
            GeneratorCacheKey::from_load_spec("z_image_turbo", &with_adapter)
        );
        assert_ne!(
            GeneratorCacheKey::from_load_spec("z_image_turbo", &with_adapter),
            GeneratorCacheKey::from_load_spec("z_image_turbo", &different_scale)
        );
    }

    #[test]
    fn cache_key_includes_control_and_ip_components() {
        let mut control = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        control.control = Some(WeightsSource::File(PathBuf::from(
            "/controls/pose.safetensors",
        )));
        let mut ip = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        ip.ip_adapter = Some(WeightsSource::Dir(PathBuf::from("/ip-adapter")));

        assert_ne!(
            GeneratorCacheKey::from_load_spec("sdxl", &control),
            GeneratorCacheKey::from_load_spec("sdxl", &ip)
        );
    }
}
