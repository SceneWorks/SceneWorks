import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Bundle only these first-party documents. Do not expose the repository docs
// directory through the development server's filesystem allow-list.
export default function editorGuidesPlugin() {
  const id = "virtual:editor-guides";
  const files = {
    scriptGuide: fileURLToPath(new URL("../../docs/film-script-writing.md", import.meta.url)),
    operatorGuide: fileURLToPath(new URL("../../docs/film-editor.md", import.meta.url)),
  };
  return {
    name: "editor-guides",
    resolveId(source) { return source === id ? `\0${id}` : null; },
    load(source) {
      if (source !== `\0${id}`) return null;
      return Object.entries(files).map(([name, path]) => {
        this.addWatchFile(path);
        return `export const ${name} = ${JSON.stringify(readFileSync(path, "utf8"))};`;
      }).join("\n");
    },
  };
}
