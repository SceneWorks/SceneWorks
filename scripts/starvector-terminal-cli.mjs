import path from "node:path";
import { pathToFileURL } from "node:url";

// `import.meta.url === `file://${process.argv[1]}`` is false on Windows because
// argv uses backslashes while file URLs use escaped forward slashes. Convert the
// resolved native path with Node's platform-aware URL helper instead.
export function isExecutedModule(metaUrl, argv1 = process.argv[1]) {
  return Boolean(argv1) && pathToFileURL(path.resolve(argv1)).href === metaUrl;
}
