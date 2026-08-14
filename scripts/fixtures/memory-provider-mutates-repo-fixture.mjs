import { writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);
if (request.action === "probe") {
  process.stdout.write(JSON.stringify({
    hardware: {
      probe: "mutation fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8"
    }
  }));
} else {
  await writeFile(path.join(request.repositoryPaths.sceneWorks, "provider-mutated.txt"), "mutation\n");
  process.stdout.write("{}");
}
