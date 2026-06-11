// JoyCaption captioner prompt corpus + prompt builder (sc-4199).
// Extracted verbatim from TrainingStudio.jsx — ~130 lines of caption prompt
// templates and the pure prompt-assembly logic that were buried in the 2.2k-line
// screen. Pure data + one pure function; no React, no app state.

export const joyCaptionModel = "fancyfeast/llama-joycaption-beta-one-hf-llava";
export const joyCaptionTypes = [
  "Descriptive",
  "Descriptive (Casual)",
  "Straightforward",
  "Stable Diffusion Prompt",
  "MidJourney",
  "Danbooru tag list",
  "e621 tag list",
  "Rule34 tag list",
  "Booru-like tag list",
  "Art Critic",
  "Product Listing",
  "Social Media Post",
];
export const joyCaptionLengths = [
  "any",
  "very short",
  "short",
  "medium-length",
  "long",
  "very long",
  "20",
  "30",
  "40",
  "50",
  "60",
  "80",
  "100",
  "120",
  "160",
  "200",
  "260",
];
export const joyCaptionExtraOptions = [
  { value: "If there is a person/character in the image you must refer to them as {name}.", label: "Use character name" },
  {
    value:
      "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    label: "Avoid fixed traits",
  },
  { value: "Include information about lighting.", label: "Include lighting" },
  { value: "Include information about camera angle.", label: "Include camera angle" },
  { value: "Do NOT include anything sexual; keep it PG.", label: "Keep it PG" },
  { value: "Do NOT mention the image's resolution.", label: "Skip resolution" },
  { value: "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.", label: "Composition style" },
  { value: "Do NOT mention any text that is in the image.", label: "Ignore text" },
  { value: "Specify the depth of field and whether the background is in focus or blurred.", label: "Depth of field" },
  { value: "Do NOT use any ambiguous language.", label: "No ambiguity" },
  { value: "ONLY describe the most important elements of the image.", label: "Important elements only" },
  { value: "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.", label: "Shot size" },
  { value: "Your response will be used by a text-to-image model, so avoid useless meta phrases like \"This image shows...\", \"You are looking at...\", etc.", label: "No meta phrases" },
];
export const joyCaptionPromptMap = {
  Descriptive: [
    "Write a detailed description for this image.",
    "Write a detailed description for this image in {word_count} words or less.",
    "Write a {length} detailed description for this image.",
  ],
  "Descriptive (Casual)": [
    "Write a descriptive caption for this image in a casual tone.",
    "Write a descriptive caption for this image in a casual tone within {word_count} words.",
    "Write a {length} descriptive caption for this image in a casual tone.",
  ],
  Straightforward: [
    'Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
    'Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
    'Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
  ],
  "Stable Diffusion Prompt": [
    "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
    "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
  ],
  MidJourney: [
    "Write a MidJourney prompt for this image.",
    "Write a MidJourney prompt for this image within {word_count} words.",
    "Write a {length} MidJourney prompt for this image.",
  ],
  "Danbooru tag list": [
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
  ],
  "e621 tag list": [
    "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
    "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
    "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
  ],
  "Rule34 tag list": [
    "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
    "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
    "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
  ],
  "Booru-like tag list": [
    "Write a list of Booru-like tags for this image.",
    "Write a list of Booru-like tags for this image within {word_count} words.",
    "Write a {length} list of Booru-like tags for this image.",
  ],
  "Art Critic": [
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
  ],
  "Product Listing": [
    "Write a caption for this image as though it were a product listing.",
    "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
    "Write a {length} caption for this image as though it were a product listing.",
  ],
  "Social Media Post": [
    "Write a caption for this image as if it were being used for a social media post.",
    "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
    "Write a {length} caption for this image as if it were being used for a social media post.",
  ],
};
export const defaultCaptionSettings = {
  captioner: "joy_caption",
  modelNameOrPath: joyCaptionModel,
  recaption: false,
  requestedGpu: "auto",
  captionType: "Descriptive",
  captionLength: "long",
  extraOptions: [],
  nameInput: "",
  temperature: "0.6",
  topP: "0.9",
  maxNewTokens: "256",
  captionPrompt: "",
  lowVram: false,
};

// Build the JoyCaption instruction prompt from the chosen type/length + extra
// options, substituting {name}/{length}/{word_count}. Pure — extracted from
// TrainingStudio (sc-4199) so the prompt corpus + builder are testable in isolation.
export function buildJoyCaptionPrompt(settings) {
  const captionLength = String(settings.captionLength || "long");
  let templateIndex = 2;
  if (captionLength === "any") {
    templateIndex = 0;
  } else if (/^\d+$/.test(captionLength)) {
    templateIndex = 1;
  }
  const templates = joyCaptionPromptMap[settings.captionType] ?? joyCaptionPromptMap.Descriptive;
  const extraOptions = Array.isArray(settings.extraOptions) ? settings.extraOptions : [];
  const prompt = [templates[templateIndex], ...extraOptions].filter(Boolean).join(" ");
  return prompt
    .replaceAll("{name}", String(settings.nameInput || "{NAME}"))
    .replaceAll("{length}", captionLength)
    .replaceAll("{word_count}", captionLength);
}
