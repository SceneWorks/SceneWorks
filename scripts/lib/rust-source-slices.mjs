/** Mask Rust comments and literals without changing offsets or line numbers. */
export function rustCodeMask(source) {
  const chars = source.split("");
  const blank = (start, end) => {
    for (let i = start; i < end; i += 1) if (chars[i] !== "\n") chars[i] = " ";
  };
  let i = 0;
  while (i < source.length) {
    const start = i;
    if (source.startsWith("//", i)) {
      i = source.indexOf("\n", i);
      if (i < 0) i = source.length;
    } else if (source.startsWith("/*", i)) {
      let depth = 1;
      i += 2;
      while (i < source.length && depth) {
        if (source.startsWith("/*", i)) { depth += 1; i += 2; }
        else if (source.startsWith("*/", i)) { depth -= 1; i += 2; }
        else i += 1;
      }
      if (depth) throw new Error("unclosed Rust block comment");
    } else {
      const raw = /^(?:br|cr|r)(#*)"/.exec(source.slice(i));
      if (raw && !/[A-Za-z0-9_]/.test(source[i - 1] ?? " ")) {
        const close = `"${raw[1]}`;
        const end = source.indexOf(close, i + raw[0].length);
        if (end < 0) throw new Error("unclosed raw Rust string");
        i = end + close.length;
      } else if (source[i] === '"') {
        i += 1;
        while (i < source.length && source[i] !== '"') i += source[i] === "\\" ? 2 : 1;
        if (i >= source.length) throw new Error("unclosed Rust string");
        i += 1;
      } else {
        const char = /^'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'/u.exec(source.slice(i));
        if (char) i += char[0].length;
        else { i += 1; continue; }
      }
    }
    blank(start, i);
  }
  return chars.join("");
}

/** Blank exact cfg(test) module items only; production after an inline test module stays graded.
 * Other cfg(test) injection items remain visible, as in the admission guard's original contract.
 */
export function withoutRustTestModules(source) {
  const code = rustCodeMask(source);
  const chars = source.split("");
  const declaration = /#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\](?:\s*#\s*\[[^\]]*\])*\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*([;{])/g;
  for (const match of code.matchAll(declaration)) {
    let end = match.index + match[0].length;
    if (match[1] === "{") {
      let depth = 1;
      while (end < code.length && depth) {
        if (code[end] === "{") depth += 1;
        if (code[end] === "}") depth -= 1;
        end += 1;
      }
      if (depth) throw new Error("unclosed cfg(test) Rust module");
    }
    for (let i = match.index; i < end; i += 1) if (chars[i] !== "\n") chars[i] = " ";
  }
  return chars.join("");
}
