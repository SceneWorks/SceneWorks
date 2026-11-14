//! Regenerate the metadata-only portable validator corpus; never loads or executes a model.
use sceneworks_core::{options, options_context, registry::builtin_model_spec};
use serde_json::{json, Value};
fn main() {
    let mut models = serde_json::Map::new();
    for (model, mode) in [
        ("flux_dev", "text_to_image"),
        ("flux3_klein", "text_to_image"),
        ("ltx_2_5", "text_to_video"),
        ("kokoro_82m", "text_to_speech"),
        ("starvector_1b", "image_to_svg"),
        ("sensenova_u1_8b", "text_to_image"),
    ] {
        models.insert(
            model.into(),
            options_context::for_model(
                builtin_model_spec(model).expect("shipped model"),
                "mlx",
                mode,
                None,
            )
            .model,
        );
    }
    let target = sceneworks_core::training::effective_training_targets_for_candle(false)
        .targets
        .into_iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .unwrap();
    models.insert(target.id.clone(), serde_json::to_value(&target).unwrap());
    let control_target = sceneworks_core::training::effective_training_targets_for_candle(true)
        .targets
        .into_iter()
        .find(|target| target.id == "krea_2_control")
        .unwrap();
    models.insert(
        control_target.id.clone(),
        serde_json::to_value(&control_target).unwrap(),
    );
    models.insert(
        "vectorPrompt".into(),
        json!({"raster":models["flux_dev"],"vector":models["starvector_1b"]}),
    );
    let image = json!({"projectId":"p","model":"flux_dev","prompt":"a tree"});
    let video = json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":"a tree"});
    let vector = json!({"projectId":"p","model":"starvector_1b","mode":"image_to_svg","sourceAssetId":"source"});
    let mut cases = vec![];
    let mut add = |name: &str, kind: &str, payload: Value| {
        let model = if kind == "vectorPrompt" {
            "vectorPrompt"
        } else {
            payload
                .get("model")
                .or_else(|| payload.get("targetId"))
                .or_else(|| payload.pointer("/plan/target/targetId"))
                .and_then(Value::as_str)
                .unwrap_or("")
        };
        let empty = serde_json::Map::new();
        let facts = models
            .get(model)
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let expected =
            match options::request_validation::resolve(kind, payload.as_object().unwrap(), facts) {
                Ok(resolved) => json!({"ok":true,"resolved":resolved}),
                Err(error) => json!({"ok":false,"refusal":error}),
            };
        cases.push(
            json!({"name":name,"kind":kind,"model":model,"payload":payload,"expected":expected}),
        );
    };
    add("image model defaults", "image", image.clone());
    for (key, value) in [
        ("width", json!("1024")),
        ("count", json!(1.5)),
        ("advanced", json!({"steps":"oops"})),
        ("advanced", json!({"steps":"20"})),
        (
            "advanced",
            json!({"lowResStart":{"version":1,"k":2,"scale":0.5}}),
        ),
        ("advanced", json!({"enhancePrompt":true})),
        ("referenceAssetIds", json!(["source", 8])),
    ] {
        let mut payload = image.clone();
        payload[key] = value.clone();
        add(&format!("image {key} {value}"), "image", payload);
    }
    add("video model defaults", "video", video.clone());
    for (key, value) in [
        ("duration", json!(30)),
        ("fps", json!(60)),
        ("referenceAudioAssetIds", json!(["audio"])),
        (
            "advanced",
            json!({"autoDuration":true,"autoDurationMinSeconds":1,"autoDurationMaxSeconds":999}),
        ),
        ("advanced", json!({"vaeDecoder":"wrong"})),
    ] {
        let mut payload = video.clone();
        payload[key] = value.clone();
        add(&format!("video {key} {value}"), "video", payload);
    }
    add("vector source", "vector", vector.clone());
    for (key, value) in [
        ("prompt", json!("unsupported guidance")),
        ("sampling", json!({"temperature":3})),
        ("detailBudget", json!({"maxNewTokens":999999})),
    ] {
        let mut payload = vector.clone();
        payload[key] = value.clone();
        add(&format!("vector {key} {value}"), "vector", payload);
    }
    add(
        "audio kokoro script",
        "audio",
        json!({"projectId":"p","model":"kokoro_82m","prompt":"hello"}),
    );
    add(
        "audio malformed steps",
        "audio",
        json!({"projectId":"p","model":"kokoro_82m","prompt":"hello","steps":"bad"}),
    );
    add(
        "VQA source question",
        "vqa",
        json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What is shown?"}),
    );
    add(
        "VQA invalid tokens",
        "vqa",
        json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What is shown?","maxNewTokens":9999}),
    );
    add(
        "interleave prompt",
        "interleave",
        json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"a tree"}),
    );
    add(
        "interleave invalid budget",
        "interleave",
        json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"a tree","maxImages":99}),
    );
    add(
        "video upscale",
        "videoUpscale",
        json!({"sourceAssetId":"clip","factor":2}),
    );
    add(
        "video upscale invalid factor",
        "videoUpscale",
        json!({"sourceAssetId":"clip","factor":3}),
    );
    add("prompt refine", "promptRefine", json!({"prompt":"a tree"}));
    add(
        "prompt refine malformed policy",
        "promptRefine",
        json!({"prompt":"a tree","kvCompression":"wrong"}),
    );
    add(
        "character test",
        "characterTest",
        json!({"prompt":"a tree"}),
    );
    add(
        "character test invalid count",
        "characterTest",
        json!({"prompt":"a tree","count":9}),
    );
    add(
        "vector composition",
        "vectorPrompt",
        json!({"projectId":"p","prompt":"a tree","rasterModel":"flux_dev","vectorModel":"starvector_1b"}),
    );
    add(
        "vector composition invalid sampler",
        "vectorPrompt",
        json!({"projectId":"p","prompt":"a tree","rasterModel":"flux_dev","vectorModel":"starvector_1b","sampling":{"topP":2}}),
    );
    add(
        "Yue2 create",
        "yue2",
        json!({"kind":"create","lyrics":"verse","computePolicy":"auto"}),
    );
    add(
        "Yue2 invalid steps",
        "yue2",
        json!({"kind":"create","lyrics":"verse","computePolicy":"auto","steps":0}),
    );
    add(
        "Yue2 execution",
        "yue2Execution",
        json!({"kind":"create","lyrics":"verse","computePolicy":"auto","runId":"yue2run_test"}),
    );
    add(
        "Yue2 execution invalid steps",
        "yue2Execution",
        json!({"kind":"create","lyrics":"verse","computePolicy":"auto","runId":"yue2run_test","steps":0}),
    );
    let training = json!({"targetId":target.id,"datasetId":"dataset","outputName":"adapter","dryRun":true,"config":target.defaults});
    add("training defaults", "training", training.clone());
    let mut invalid_training = training.clone();
    invalid_training["config"]["rank"] = json!(129);
    add("training target limit", "training", invalid_training);
    let plan = json!({"plan":{"target":{"targetId":target.id,"kernel":target.kernel,"baseModel":target.base_model},"config":target.defaults}});
    add("lora_train defaults", "lora_train", plan.clone());
    let mut invalid = plan.clone();
    invalid["plan"]["config"]["rank"] = json!(129);
    add("lora_train target limit", "lora_train", invalid);
    add("control_training wrong kernel", "control_training", plan);
    let control = json!({"plan":{"target":{"targetId":control_target.id,"kernel":control_target.kernel,"baseModel":control_target.base_model},"config":control_target.defaults}});
    add(
        "control_training defaults",
        "control_training",
        control.clone(),
    );
    let mut invalid = control.clone();
    invalid["plan"]["config"]["batchSize"] = json!(0);
    add(
        "control_training invalid batch",
        "control_training",
        invalid,
    );
    println!("{}",serde_json::to_string_pretty(&json!({"schemaVersion":options::request_validation::VALIDATOR_SCHEMA_VERSION,"modelFacts":models,"cases":cases})).unwrap());
}
