// Parser for the authored style catalog at documents/style.txt → the structured
// styles.json this directory ships. Kept as a pure function (no fs) so both the
// generator script (scripts/generate-styles.mjs) and the drift-guard test
// (styleCatalog.test.js) run the exact same derivation — styles.json can never
// hand-drift away from style.txt. See the accents.js / theme-init precedent.
//
// Source format: one style per line. Top-level group lines start at column 0;
// their fine-grained styles are indented one tab. Each line is
// "Name: fine-tuned description text". A handful of names contain their own
// colon (e.g. "Cyberpunk: Edgerunners Style") — the name regex tolerates one
// extra short "…: Xxx Style" segment before the description colon.

export const PROMPT_TEMPLATE = "Style: {style}\nDescription: {description}";
export const CATALOG_VERSION = 1;
export const CATALOG_SOURCE = "documents/style.txt";

const LINE_RE = /^(?<name>[^:]+(?:: [^:]{1,40} Style)?):\s*(?<desc>.+)$/;

// NFKD-decompose, drop combining marks, then collapse every non-alphanumeric run
// to a single dash. Mirrors the reference Python (NFKD + ascii-ignore): "Comic
// Européen Prestige" → "comic-europeen-prestige".
export function slugify(name) {
  return name
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function parseLine(raw) {
  const m = LINE_RE.exec(raw.trim());
  if (!m) {
    throw new Error(`unparseable style line: ${JSON.stringify(raw.slice(0, 80))}`);
  }
  return { name: m.groups.name.trim(), desc: m.groups.desc.trim() };
}

/**
 * Parse the raw style.txt text into the structured catalog object.
 * @param {string} rawText contents of documents/style.txt
 * @returns {{version:number, source:string, promptTemplate:string, groups:Array}}
 */
export function parseStyleCatalog(rawText) {
  const groups = [];
  const seen = new Map(); // slug -> entry (for dedup / rename)
  let current = null;

  const lines = rawText.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    if (!raw.trim()) continue; // skip blank lines

    const isSub = raw.startsWith("\t");
    const { name, desc } = parseLine(raw);

    if (!isSub) {
      current = { id: slugify(name), name, description: desc, styles: [] };
      groups.push(current);
      continue;
    }

    if (!current) {
      throw new Error(`line ${i + 1}: sub-style before any group`);
    }

    let slug = slugify(name);
    let displayName = name;
    if (seen.has(slug)) {
      const prev = seen.get(slug);
      if (prev.prompt === desc) {
        continue; // identical duplicate — drop it
      }
      // Same name, genuinely different text: keep it as "Name 2", "Name 3", …
      let n = 2;
      while (seen.has(`${slug}-${n}`)) n += 1;
      displayName = `${name} ${n}`;
      slug = `${slug}-${n}`;
    }

    const entry = { id: slug, name: displayName, prompt: desc };
    seen.set(slug, entry);
    current.styles.push(entry);
  }

  return {
    version: CATALOG_VERSION,
    source: CATALOG_SOURCE,
    promptTemplate: PROMPT_TEMPLATE,
    groups,
  };
}
