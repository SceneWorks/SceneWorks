//! A registered trainer whose base weights load **on first need** (sc-2124, epic 2123 E3): the
//! descriptor and every weights-free `validate` floor answer without reading a weight, and `train`
//! runs those floors — the technique refusals included — before the base loads, so an unsupported
//! request is refused with nothing loaded, cached or written. The family supplies its load and its
//! floors; everything else is identical across the MLX LoRA trainers.

use std::cell::OnceCell;

use gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};

use crate::Result;

/// The family's weights-free validate floors (everything its loaded trainer's `validate` checks
/// that does not need the loaded base).
pub type ValidateFloors = fn(&TrainerDescriptor, &TrainingRequest) -> gen_core::Result<()>;

/// A [`Trainer`] that loads its base `T` on first need. See the module docs.
pub struct LazyTrainer<T: Trainer> {
    descriptor: TrainerDescriptor,
    floors: ValidateFloors,
    needs_base: fn(&TrainingRequest) -> bool,
    load: Box<dyn Fn() -> Result<T>>,
    loaded: OnceCell<T>,
}

impl<T: Trainer> LazyTrainer<T> {
    /// A trainer advertising `descriptor` whose `validate` runs `floors` until the base is loaded,
    /// and that runs `load` the first time a base is needed.
    pub fn new(
        descriptor: TrainerDescriptor,
        floors: ValidateFloors,
        load: impl Fn() -> Result<T> + 'static,
    ) -> Self {
        Self {
            descriptor,
            floors,
            needs_base: |_| false,
            load: Box::new(load),
            loaded: OnceCell::new(),
        }
    }

    /// Validate requests for which `needs_base` holds against the loaded base (e.g. custom
    /// `lora_target_modules`, which only the loaded model can match) — loading it if needed.
    pub fn validating_on_base_when(mut self, needs_base: fn(&TrainingRequest) -> bool) -> Self {
        self.needs_base = needs_base;
        self
    }

    fn loaded(&self) -> Result<&T> {
        if self.loaded.get().is_none() {
            let _ = self.loaded.set((self.load)()?);
        }
        Ok(self.loaded.get().expect("the base was loaded above"))
    }
}

/// Custom `lora_target_modules` — the request only the loaded base can validate.
pub fn custom_targets(req: &TrainingRequest) -> bool {
    !req.config.lora_target_modules.is_empty()
}

impl<T: Trainer> Trainer for LazyTrainer<T> {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if let Some(trainer) = self.loaded.get() {
            return trainer.validate(req);
        }
        if (self.needs_base)(req) {
            return self.loaded().map_err(gen_core::Error::from)?.validate(req);
        }
        (self.floors)(&self.descriptor, req)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Every validate floor (techniques included, epic 2123 E3) refuses before the base loads.
        self.validate(req)?;
        self.loaded().map_err(gen_core::Error::from)?;
        let trainer = self.loaded.get_mut().expect("the base was loaded above");
        trainer.train(req, on_progress)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::rc::Rc;

    use gen_core::train::{TrainingConfig, TrainingItem, TrainingTechniques};
    use gen_core::Modality;

    use super::*;

    const DESC: TrainerDescriptor = TrainerDescriptor {
        id: "lazy_probe",
        family: "probe",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: TrainingTechniques::NONE,
    };

    struct Loaded;
    impl Trainer for Loaded {
        fn descriptor(&self) -> &TrainerDescriptor {
            &DESC
        }
        fn validate(&self, _req: &TrainingRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn train(
            &mut self,
            _req: &TrainingRequest,
            _on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> gen_core::Result<TrainingOutput> {
            Err(gen_core::Error::Msg("trained".into()))
        }
    }

    fn lazy(loads: &Rc<Cell<u32>>) -> LazyTrainer<Loaded> {
        let loads = loads.clone();
        LazyTrainer::new(
            DESC,
            gen_core::train::validate_training_techniques,
            move || {
                loads.set(loads.get() + 1);
                Ok(Loaded)
            },
        )
    }

    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(PathBuf::from("a.png"), "a".into())],
            config: TrainingConfig::default(),
            output_dir: PathBuf::from("/out"),
            file_name: "x.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        }
    }

    /// The floors answer `validate` and refuse at `train` with nothing loaded; an accepted `train`
    /// loads once; `needs_base` routes validate through the base. Mutations: load before the
    /// floors in `train` ⇒ `loads == 1` after the refusal ⇒ red; ignore `needs_base` ⇒ red.
    #[test]
    fn floors_refuse_before_the_base_loads_and_the_base_loads_once() {
        let loads = Rc::new(Cell::new(0));
        let mut t = lazy(&loads);
        let mut noisy = request();
        noisy.config.weight_noise_sigma = 0.01;
        assert!(matches!(
            t.validate(&noisy),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            t.train(&noisy, &mut |_| {}),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert_eq!(loads.get(), 0, "a refused request loaded the base");
        let err = t.train(&request(), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("trained"), "{err}");
        let _ = t.train(&request(), &mut |_| {});
        assert_eq!(loads.get(), 1);

        let loads = Rc::new(Cell::new(0));
        let t = lazy(&loads).validating_on_base_when(custom_targets);
        t.validate(&request()).unwrap();
        assert_eq!(loads.get(), 0);
        let mut custom = request();
        custom.config.lora_target_modules = vec!["to_q".into()];
        t.validate(&custom).unwrap();
        assert_eq!(loads.get(), 1, "custom targets validate on the loaded base");
    }
}
