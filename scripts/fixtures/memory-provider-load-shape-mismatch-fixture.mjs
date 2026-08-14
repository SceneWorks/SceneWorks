import process from "node:process";

// Attests the planned strategy exactly but the OPPOSITE materialization shape. Eager and deferred
// measurements are not interchangeable, so the runner must reject this capture rather than adopt
// the plan's declared shape as the receipt.
let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);
if (request.action === "probe") {
  process.stdout.write(JSON.stringify({
    hardware: {
      probe: "executable fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8"
    }
  }));
} else {
  process.stdout.write(JSON.stringify({
    strategy: request.planned.strategy,
    loadShape: request.planned.loadShape === "eager_materialization"
      ? "deferred_materialization"
      : "eager_materialization",
  }));
}
