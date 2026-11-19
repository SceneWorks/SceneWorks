//! Regenerate the metadata-only portable validator corpus; never loads or executes a model.
use sceneworks_core::{options, options_context, registry::builtin_model_spec};
use serde_json::{json, Value};
fn main() {
    let mut models = serde_json::Map::new();
    for (model, mode) in [
        ("flux_dev", "text_to_image"),
        ("qwen_image_2_1", "text_to_image"),
        ("flux3_klein", "text_to_image"),
        ("ltx_2_5", "text_to_video"),
        ("wan_2_2_i2v_14b", "first_last_frame"),
        ("kokoro_82m", "text_to_speech"),
        ("starvector_1b", "image_to_svg"),
        ("sensenova_u1_8b", "text_to_image"),
        ("scail2_14b", "animate_character"),
        ("svd", "image_to_video"),
        ("moss_image_base", "text_to_image"),
        ("moss_image_sft", "text_to_image"),
        ("supra2_img", "text_to_image"),
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
    add(
        "image default caption policy",
        "image",
        json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"captionOverflow":" default "}}),
    );
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
    add(
        "SVD promptless image animation",
        "video",
        json!({"projectId":"p","model":"svd","mode":"image_to_video","prompt":"","sourceAssetId":"image"}),
    );
    add(
        "ordinary video blank prompt",
        "video",
        json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":""}),
    );
    add(
        "SVD promptless unsupported mode",
        "video",
        json!({"projectId":"p","model":"svd","mode":"text_to_video","prompt":""}),
    );

    add(
        "video unsupported backend mode",
        "video",
        json!({"projectId":"p","model":"wan_2_2_i2v_14b","mode":"first_last_frame","prompt":"a tree","sourceAssetId":"first","lastFrameAssetId":"last"}),
    );
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
    add(
        "image model stride refusal",
        "image",
        json!({"projectId":"p","model":"qwen_image_2_1","prompt":"tree","width":1001,"height":1024}),
    );
    add(
        "image FLUX3 reference cap",
        "image",
        json!({"projectId":"p","model":"flux3_klein","prompt":"tree","referenceAssetIds":(0..11).map(|i| format!("source-{i}")).collect::<Vec<_>>()}),
    );
    add(
        "video LTX extend mode refusal",
        "video",
        json!({"projectId":"p","model":"ltx_2_5","mode":"extend_clip","prompt":"tree","sourceClipAssetId":"clip"}),
    );
    add(
        "video reference cap",
        "video",
        json!({"projectId":"p","model":"ltx_2_5","mode":"reference_to_video","prompt":"tree","referenceAssetIds":(0..9).map(|i| format!("source-{i}")).collect::<Vec<_>>()}),
    );
    add(
        "video audio-only references",
        "video",
        json!({"projectId":"p","model":"ltx_2_5","mode":"reference_to_video","prompt":"tree","referenceAudioAssetIds":["audio"]}),
    );
    add(
        "video SCAIL character cap",
        "video",
        json!({"projectId":"p","model":"scail2_14b","mode":"animate_character","prompt":"tree","sourceClipAssetId":"clip","referenceAssetIds":(0..7).map(|i| format!("source-{i}")).collect::<Vec<_>>()}),
    );
    add(
        "video SCAIL paired descriptor",
        "video",
        json!({"projectId":"p","model":"scail2_14b","mode":"animate_character","prompt":"tree","sourceClipAssetId":"clip","referenceAssetIds":["person1","person2"]}),
    );
    for model in ["moss_image_base", "moss_image_sft", "supra2_img"] {
        add(
            &format!("{model} bare published defaults"),
            "image",
            json!({"projectId":"p","model":model,"prompt":"tree"}),
        );
    }
    for (name, payload) in [
        (
            "MOSS SFT DMD defaults",
            json!({"projectId":"p","model":"moss_image_sft","prompt":"tree","advanced":{"mossImage":{"acceleration":"unmerged_bf16"}}}),
        ),
        (
            "MOSS Base DMD refusal",
            json!({"projectId":"p","model":"moss_image_base","prompt":"tree","advanced":{"mossImage":{"acceleration":"unmerged_bf16"}}}),
        ),
        (
            "MOSS flash attention refusal",
            json!({"projectId":"p","model":"moss_image_sft","prompt":"tree","advanced":{"flashAttn":false}}),
        ),
        (
            "MOSS edit defaults",
            json!({"projectId":"p","model":"moss_image_sft","mode":"edit_image","prompt":"tree","referenceAssetIds":["source"]}),
        ),
        (
            "MOSS malformed nested options",
            json!({"projectId":"p","model":"moss_image_sft","prompt":"tree","advanced":{"mossImage":{"uncond_skip_layers":"bad"}}}),
        ),
        (
            "MOSS positive embedding prompt omission",
            json!({"projectId":"p","model":"moss_image_sft","mode":"edit_image","prompt":"","referenceAssetIds":["source"],"advanced":{"mossImageAdvanced":{"positive":{"values":{"assetId":"embedding","sha256":"00"}}}}}),
        ),
        (
            "Supra2 unsupported reference",
            json!({"projectId":"p","model":"supra2_img","prompt":"tree","referenceAssetIds":["source"]}),
        ),
        (
            "Supra2 negative guidance allowed",
            json!({"projectId":"p","model":"supra2_img","prompt":"tree","advanced":{"guidanceScale":-1}}),
        ),
        (
            "Supra2 unsupported negative prompt",
            json!({"projectId":"p","model":"supra2_img","prompt":"tree","negativePrompt":"bad"}),
        ),
        (
            "FLUX PAG valid options",
            json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"pag":{"scale":1.5,"layers":[2,5],"start":0.1,"stop":0.85}}}),
        ),
        (
            "FLUX PAG out-of-range block refusal",
            json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"pag":{"scale":1.5,"layers":[99]}}}),
        ),
        (
            "Supra2 PAG unsupported refusal",
            json!({"projectId":"p","model":"supra2_img","prompt":"tree","advanced":{"pag":{"scale":1.5}}}),
        ),
    ] {
        add(name, "image", payload);
    }
    for (name, value) in [
        (
            "ImageDetail authored tile repository",
            json!("example/control"),
        ),
        ("ImageDetail invalid tile repository", json!(42)),
    ] {
        add(
            name,
            "imageDetail",
            json!({"projectId":"p","model":"flux_dev","sourceAssetId":"source","advanced":{"tileControlNetRepo":value}}),
        );
    }
    add(
        "MOSS top-level positive tensor carrier",
        "image",
        json!({"projectId":"p","model":"moss_image_sft","mode":"edit_image","prompt":"","referenceAssetIds":["source"],"mossImageAdvanced":{"positive":{"values":{"assetId":"embedding","sha256":"00"}}}}),
    );
    add(
        "MOSS duplicate tensor carriers refuse",
        "image",
        json!({"projectId":"p","model":"moss_image_sft","mode":"edit_image","prompt":"tree","referenceAssetIds":["source"],"mossImageAdvanced":{},"advanced":{"mossImageAdvanced":{}}}),
    );
    for (name, value) in [
        ("utility authored tier", json!("4")),
        ("utility invalid tier", json!("bad")),
    ] {
        add(name, "utility", json!({"advanced":{"mlxQuantize":value}}));
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
        "VQA clamped tokens",
        "vqa",
        json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What is shown?","maxNewTokens":9999}),
    );
    add(
        "interleave prompt",
        "interleave",
        json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"a tree"}),
    );
    add(
        "interleave clamped top-level budget",
        "interleave",
        json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"a tree","maxImages":99}),
    );
    for (name, kind, payload) in [
        (
            "VQA legacy numeric controls",
            "vqa",
            json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What?","maxNewTokens":"4096","maxImagePixels":2.5}),
        ),
        (
            "VQA historical invalid integer fallback",
            "vqa",
            json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What?","maxNewTokens":"bad"}),
        ),
        (
            "interleave legacy numeric controls",
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"9999","height":12.9,"advanced":{"timestepShift":1.5,"schedulerShift":9,"maxNewTokens":"9000","numInferenceSteps":0}}),
        ),
        (
            "interleave historical invalid geometry fallback",
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"bad"}),
        ),
    ] {
        add(name, kind, payload);
    }
    for spelling in ["2.5", "1e100", "bad"] {
        add(
            &format!("interleave historical integer string fallback {spelling}"),
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","maxImages":spelling}),
        );
    }
    add(
        "interleave historical geometry strings",
        "interleave",
        json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"2.5","height":"1e100","maxImages":null}),
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
    for kind in ["personDetect", "personTrack"] {
        let mut valid = json!({"sourceAssetId":"clip","confidence":0.7});
        if kind == "personTrack" {
            valid["representativeFrameAssetId"] = json!("frame");
            valid["detection"] = json!({"id":"person-1"});
        }
        add(&format!("{kind} legacy confidence"), kind, valid.clone());
        let mut invalid = valid.clone();
        invalid["advanced"] = json!({"confidence":"bad"});
        add(&format!("{kind} invalid confidence"), kind, invalid);
        add(&format!("{kind} missing source"), kind, json!({}));
        for (key, value) in [("sourceAssetId", Value::Null), ("advanced", Value::Null)] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            add(&format!("{kind} null {key}"), kind, invalid);
        }
        if kind == "personTrack" {
            for (key, value) in [
                ("representativeFrameAssetId", json!("")),
                ("detection", json!({})),
                ("trackName", Value::Null),
            ] {
                let mut invalid = valid.clone();
                invalid[key] = value;
                add(&format!("{kind} invalid {key}"), kind, invalid);
            }
        }
    }
    for (name, payload) in [
        (
            "vision blank paths",
            json!({"task":"image_caption","imagePaths":["",42]}),
        ),
        (
            "vision nonstring path",
            json!({"task":"image_caption","imagePaths":["image.png",42]}),
        ),
        (
            "vision null paths",
            json!({"task":"image_caption","imagePaths":null}),
        ),
        (
            "vision path cap",
            json!({"task":"image_caption","imagePaths":vec!["image.png";7]}),
        ),
        (
            "vision null prompt",
            json!({"task":"image_caption","sourceAssetId":"source","prompt":null}),
        ),
        (
            "null references",
            json!({"prompt":"tree","sourceAssetIds":null}),
        ),
        (
            "magic defaults",
            json!({"prompt":"tree","task":"magic_prompt"}),
        ),
        (
            "describe defaults",
            json!({"task":"image_describe","sourceAssetId":"source"}),
        ),
        (
            "rewrite defaults",
            json!({"prompt":"tree","task":"qwen_image_rewrite"}),
        ),
    ] {
        add(name, "promptRefine", payload);
    }
    add(
        "image worker-owned enhancement receipt",
        "image",
        json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"promptEnhancement":null}}),
    );
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
