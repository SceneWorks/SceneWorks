import { describe, it, expect } from "vitest";
import { jobOutputsPending } from "./jobTypes.js";
import { jobClearsFromStudio, bucketFor, jobSub } from "./simple/SimpleJobCard.jsx";
import { selectStackedJobs } from "./components/generationStudio.jsx";
describe("durable output publication readiness", () => {
  it("retains a pending completed render with actionable publication status", () => {
    const job = { id: "job_pending", status: "completed", result: { outputPublication: { state: "pending" } } };
    expect(jobOutputsPending(job)).toBe(true);
    expect(jobClearsFromStudio(job)).toBe(false);
    expect(bucketFor(job)).toBe("running");
    expect(jobSub(job)).toMatch(/Publishing outputs.*job_pending/);
    expect(selectStackedJobs([job], () => true)).toEqual([job]);
    const ready = { ...job, result: { assets: [{ id: "output" }] } };
    expect(jobOutputsPending(ready)).toBe(false);
    expect(jobClearsFromStudio(ready)).toBe(true);
    expect(selectStackedJobs([ready], () => true)).toEqual([]);
  });
});
