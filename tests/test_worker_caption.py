from __future__ import annotations

from worker_runtime_shared import *

def test_joy_caption_prompt_builder_applies_length_and_name_options():
    options = JoyCaptionOptions(
        caption_type="Descriptive",
        caption_length="40",
        extra_options=["If there is a person/character in the image you must refer to them as {name}."],
        name_input="Mira",
    )

    prompt = build_joy_caption_prompt(options)

    assert "40 words or less" in prompt
    assert "Mira" in prompt

def test_caption_with_trigger_words_prepends_missing_tokens():
    caption = caption_with_trigger_words("studio portrait with soft light", ["miraStyle", "studio"])

    assert caption == "miraStyle, studio portrait with soft light"

def test_normalize_processor_resample_replaces_unsupported_lanczos():
    processor = SimpleNamespace(image_processor=SimpleNamespace(resample="lanczos"))

    normalize_processor_resample(processor)

    assert processor.image_processor.resample == JOY_CAPTION_RESAMPLE

