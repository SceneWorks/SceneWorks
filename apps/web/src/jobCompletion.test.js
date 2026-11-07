import { describe, it, expect } from "vitest";
import { jobClearsFromStudio, bucketFor, jobSub } from "./simple/SimpleJobCard.jsx";
import { selectStackedJobs } from "./components/generationStudio.jsx";

describe("shared job completion", () => {
  it.each(["image", "tensor"])("completes a native %s batch using ordinary terminal status", (type) => {
    const job = { id: "native-batch", type: "image", status: "completed", result: { assets: [{ id: "first", type }, { id: "second", type }] } };
    expect(jobClearsFromStudio(job)).toBe(true);
    expect(bucketFor(job)).toBe("done");
    expect(jobSub(job)).toBe("image");
    expect(selectStackedJobs([job], () => true)).toEqual([]);
  });
  it("keeps unfinished native jobs visible until their terminal event", () => {
    const job = { id: "native-batch", status: "running" };
    expect(jobClearsFromStudio(job)).toBe(false);
    expect(bucketFor(job)).toBe("running");
    expect(selectStackedJobs([job], () => true)).toEqual([job]);
  });
});
