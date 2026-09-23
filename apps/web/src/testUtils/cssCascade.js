// Enough of the CSS cascade to answer "which declaration actually wins here?" (sc-24111).
//
// Why this exists: jsdom applies no stylesheet and does no layout, so `getComputedStyle` sees
// nothing — a stylesheet assertion in this repo has to be made against the source text. The first
// version of the transparency-backdrop test grepped for substrings, and a grep cannot tell the
// difference between "this rule sets a checkerboard" and "this rule sets a checkerboard and a
// later, more specific rule sets it back to a flat colour". That second case is not hypothetical:
// it is exactly how `.preview-modal img { width: auto }` was silently losing to
// `.preview-modal-stage img { width: 100% }`.
//
// So: parse rules, compute specificity, and resolve a property the way a browser would — by
// (specificity, document order) over the rules that would match a described element.
//
// # Scope, stated so nobody mistakes it for a CSS engine
//
// * Selectors are descendant chains of compounds (`.a .b img`). Child/sibling combinators are not
//   modelled; a selector containing one is treated as non-matching, which is conservative in the
//   direction that matters only if someone introduces one — see `UNMODELLED_COMBINATORS`.
// * An element is described by its tag, its own classes and its ancestors' classes. A selector
//   matches when its last compound fits the element and every other compound's classes are in the
//   ancestor set. Order along the chain is not checked, which can only ever make MORE rules
//   candidates, so a re-flattening rule cannot hide from the resolver.
// * `@media` and `@supports` blocks are flattened in place rather than evaluated. A rule inside one
//   therefore competes on equal terms, which is again the conservative direction: a media query
//   that re-flattens a surface fails the test instead of being skipped.
// * Other at-rules (`@keyframes`, `@font-face`, `@property`) are skipped entirely.

/** Combinators this resolver does not model. A selector containing one never matches. */
const UNMODELLED_COMBINATORS = [">", "+", "~"];

/** Strip `/* ... *\/` comments without disturbing offsets that matter (none do). */
function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, "");
}

/**
 * Flatten a stylesheet into `{ selectors, body, order }` rules, descending into `@media` and
 * `@supports` and skipping every other at-rule.
 */
export function parseRules(text, { source = "", startOrder = 0 } = {}) {
  const css = stripComments(text);
  const rules = [];
  let order = startOrder;

  const walk = (start, end) => {
    let index = start;
    let preludeStart = index;
    while (index < end) {
      const char = css[index];
      if (char === "}") {
        index += 1;
        preludeStart = index;
        continue;
      }
      if (char !== "{") {
        index += 1;
        continue;
      }
      const prelude = css.slice(preludeStart, index).trim();
      // Find the matching close brace.
      let depth = 1;
      let cursor = index + 1;
      while (cursor < end && depth > 0) {
        if (css[cursor] === "{") depth += 1;
        else if (css[cursor] === "}") depth -= 1;
        cursor += 1;
      }
      const bodyStart = index + 1;
      const bodyEnd = cursor - 1;
      if (prelude.startsWith("@")) {
        if (/^@(media|supports|layer|container)\b/.test(prelude)) {
          walk(bodyStart, bodyEnd);
        }
      } else if (prelude) {
        rules.push({
          source,
          order: order++,
          selectors: prelude
            .split(",")
            .map((selector) => selector.trim())
            .filter(Boolean),
          body: css.slice(bodyStart, bodyEnd),
        });
      }
      index = cursor;
      preludeStart = index;
    }
  };

  walk(0, css.length);
  return { rules, nextOrder: order };
}

/** CSS specificity as a comparable tuple `[id, class, type]`. */
export function specificity(selector) {
  const withoutPseudoElements = selector.replace(/::[a-zA-Z-]+/g, "");
  const ids = withoutPseudoElements.match(/#[\w-]+/g)?.length ?? 0;
  const classes =
    (withoutPseudoElements.match(/\.[\w-]+/g)?.length ?? 0) +
    (withoutPseudoElements.match(/\[[^\]]+\]/g)?.length ?? 0) +
    (withoutPseudoElements.match(/:(?!:)[a-zA-Z-]+/g)?.length ?? 0);
  const types =
    withoutPseudoElements
      .replace(/\.[\w-]+/g, " ")
      .replace(/#[\w-]+/g, " ")
      .replace(/\[[^\]]+\]/g, " ")
      .replace(/:(?!:)[a-zA-Z-]+(\([^)]*\))?/g, " ")
      .split(/\s+/)
      .filter((token) => /^[a-zA-Z][\w-]*$/.test(token)).length;
  return [ids, classes, types];
}

const compare = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

/** Split a compound like `img.foo.bar` into `{ tag, classes }`. */
function compound(text) {
  const classes = (text.match(/\.[\w-]+/g) ?? []).map((token) => token.slice(1));
  const tag = /^[a-zA-Z][\w-]*/.exec(text)?.[0]?.toLowerCase() ?? null;
  return { tag, classes };
}

/**
 * Would `selector` match an element described by `{ tag, classes, ancestorClasses }`?
 *
 * See the scope note at the top: this is deliberately permissive about chain order, because a
 * false MATCH only adds a competitor to the cascade (making the test stricter) while a false miss
 * would let a re-flattening rule escape.
 */
export function selectorMatches(selector, element) {
  if (UNMODELLED_COMBINATORS.some((combinator) => selector.includes(combinator))) return false;
  if (selector.includes("::")) return false;
  const compounds = selector.split(/\s+/).filter(Boolean).map(compound);
  if (compounds.length === 0) return false;
  const own = new Set(element.classes ?? []);
  const ancestors = new Set(element.ancestorClasses ?? []);
  const last = compounds[compounds.length - 1];
  if (last.tag && last.tag !== element.tag) return false;
  if (!last.classes.every((name) => own.has(name))) return false;
  return compounds
    .slice(0, -1)
    .every(({ tag, classes }) => !tag && classes.every((name) => ancestors.has(name)));
}

/** Every `prop: value` declaration in a rule body, last-wins within the rule. */
function declarations(body) {
  const out = new Map();
  for (const piece of body.split(";")) {
    const at = piece.indexOf(":");
    if (at < 0) continue;
    const property = piece.slice(0, at).trim().toLowerCase();
    if (!property || property.startsWith("--")) continue;
    out.set(property, piece.slice(at + 1).trim());
  }
  return out;
}

/**
 * The declaration that WINS for `property` on `element`, resolved over `rules`.
 *
 * `expand` maps a shorthand onto the longhand being asked for, because the whole point here is
 * that `background: var(--bg-2)` resets `background-image` — a resolver that only looked at
 * `background-image` declarations would report the checkerboard as the winner while the browser
 * paints a flat colour.
 */
export function winningDeclaration(rules, element, property, expand = () => undefined) {
  let winner = null;
  for (const rule of rules) {
    const matching = rule.selectors.filter((selector) => selectorMatches(selector, element));
    if (matching.length === 0) continue;
    const best = matching
      .map(specificity)
      .reduce((a, b) => (compare(a, b) >= 0 ? a : b));
    const declared = declarations(rule.body);
    for (const [name, value] of declared) {
      const resolved = name === property ? value : expand(name, value, property);
      if (resolved === undefined) continue;
      const candidate = {
        rule,
        selector: matching[0],
        specificity: best,
        value: resolved,
        via: name,
      };
      if (
        !winner ||
        compare(best, winner.specificity) > 0 ||
        (compare(best, winner.specificity) === 0 && rule.order >= winner.rule.order)
      ) {
        winner = candidate;
      }
    }
  }
  return winner;
}

/** `background: <x>` sets `background-image` to the gradient it names, or to `none`. */
export function backgroundShorthand(name, value, property) {
  if (property !== "background-image" || name !== "background") return undefined;
  return /gradient|url\(|image-set/.test(value) ? value : "none";
}
