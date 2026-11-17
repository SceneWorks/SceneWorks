//! Character Test and prompt-authoring requests have their own public defaults.
//! Their route qualifiers preserve those defaults when the same field occurs in Image Studio.
use crate::{DefaultRule::*, IntentClass::*, OptionSpec, Predicate::*, Shape::*};

macro_rules! field {
    ($lane:literal, $key:literal, $shape:expr, $default:expr, $intent:expr, $binding:literal) => {
        OptionSpec::new(
            concat!($lane, ".", $key),
            $lane,
            $key,
            $shape,
            $default,
            $intent,
            $binding,
        )
    };
}

pub(crate) const SPECS: &[OptionSpec] = &[
    field!(
        "characterTest",
        "prompt",
        Text,
        Required,
        WhatToMake,
        "ImageRequest.prompt"
    )
    .applies(Modality("image"))
    .limits(&["1..=4000 characters"]),
    field!(
        "characterTest",
        "model",
        Text,
        Literal(r#""z_image_turbo""#),
        WhatToMake,
        "ImageRequest.model"
    )
    .applies(Modality("image")),
    field!(
        "characterTest",
        "count",
        Integer,
        Literal("4"),
        WhatToMake,
        "ImageRequest.count"
    )
    .applies(Modality("image"))
    .limits(&["1..=8"]),
    field!(
        "characterTest",
        "width",
        Integer,
        Literal("1024"),
        WhatToMake,
        "ImageRequest.width"
    )
    .applies(Modality("image"))
    .limits(&["256..=4096"]),
    field!(
        "characterTest",
        "height",
        Integer,
        Literal("1024"),
        WhatToMake,
        "ImageRequest.height"
    )
    .applies(Modality("image"))
    .limits(&["256..=4096"]),
    field!(
        "characterTest",
        "requestedGpu",
        Text,
        Literal(r#""auto""#),
        MachineBudget,
        "Job.requested_gpu"
    )
    .applies(Modality("image")),
    field!(
        "characterTest",
        "lookId",
        Text,
        Absent,
        WhatToMake,
        "ImageRequest.character_look_id"
    )
    .applies(Modality("image")),
    field!(
        "promptRefine",
        "prompt",
        Text,
        Literal(r#""""#),
        WhatToMake,
        "TextLlm authored prompt"
    )
    .limits(&["nonempty unless vision task"]),
    field!(
        "promptRefine",
        "modelId",
        Text,
        Absent,
        WhatToMake,
        "target model prompt guidance"
    ),
    field!(
        "promptRefine",
        "workflow",
        Text,
        Literal(r#""image""#),
        WhatToMake,
        "prompt rewrite workflow"
    ),
    field!(
        "promptRefine",
        "guide",
        Text,
        Absent,
        WhatToMake,
        "prompt rewrite system guide"
    ),
    field!(
        "promptRefine",
        "task",
        Text,
        Literal(r#""refine""#),
        WhatToMake,
        "prompt-refine task dispatch"
    )
    .limits(&["prompts::refine_prompt task selector"]),
    field!(
        "promptRefine",
        "aspectRatio",
        Text,
        Literal(r#""1:1""#),
        WhatToMake,
        "magic-prompt structured layout ratio"
    ),
    field!(
        "promptRefine",
        "sourceAssetId",
        Text,
        Absent,
        WhatToMake,
        "vision conditioning image"
    )
    .limits(&["project-owned image"]),
    field!(
        "promptRefine",
        "sourceAssetIds",
        Strings,
        Literal("[]"),
        WhatToMake,
        "ordered vision conditioning images"
    )
    .limits(&["MAX_MOOD_BOARD_IMAGES / MAX_QWEN_REWRITE_IMAGES"])
    .compat("Nonempty plural selection overrides singular sourceAssetId."),
    field!(
        "promptRefine",
        "projectId",
        Text,
        Absent,
        Context,
        "project scope for source images"
    ),
    field!(
        "promptRefine",
        "model",
        Text,
        Absent,
        WhatToMake,
        "TextLlm repository"
    ),
    field!(
        "promptRefine",
        "thinkingMode",
        OneOf("disabled | enabled | auto"),
        Absent,
        WhatToMake,
        "film-plan reasoning control"
    ),
    field!(
        "promptRefine",
        "captionStyle",
        OneOf("prose | tags"),
        Literal(r#""prose""#),
        WhatToMake,
        "image-describe output style"
    )
    .compat("Unknown or absent uses prose."),
    field!(
        "promptRefine",
        "kvCompression",
        OneOf("off | qualified"),
        Provider("worker SCENEWORKS_LLM_KV_COMPRESSION"),
        MachineBudget,
        "TextLlm KV compression policy"
    ),
];
