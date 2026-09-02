import process from "node:process";

let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);
if (request.action === "probe") {
  process.stdout.write(JSON.stringify({
    hardware: {
      probe: "composition mismatch fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8"
    }
  }));
} else {
  // A canonical, individually VALID composition that is never the one asked for, so the runner's
  // planned-vs-measured comparison is what rejects it rather than rung-order validation.
  const mismatch = request.planned.strategy.rung === "bounded_decode"
    ? { rung: "resident", engagedRungs: ["resident"] }
    : { rung: "bounded_decode", engagedRungs: ["resident", "bounded_decode"] };
  process.stdout.write(JSON.stringify({
    strategy: { ...request.planned.strategy, ...mismatch },
  }));
}
