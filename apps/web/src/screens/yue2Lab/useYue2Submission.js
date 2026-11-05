import { useState } from "react";
import { writeLicenseAck } from "../../licenseAcknowledgment.js";
import { buildYue2JobRequest, yue2RequestProblems } from "../../yue2Lab.js";
import { submitYue2Jobs } from "../../api/yue2.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2Submission({
  model,
  setAck,
  settings,
  problemContext,
  transcriptionReviewProblem,
  requestedGpu,
  projectId,
  token,
  setSubmitted
}) {
  const [submitError, setSubmitError] = useState(null);

  const [submitting, setSubmitting] = useState(false);

  // The server says the acceptance lapsed (the terms changed, or it was withdrawn): re-gate.
  function licenseLapsed() {
    writeLicenseAck(model.id, false);
    setAck({ status: "ready", acknowledged: false, error: null });
  }

  async function submit(kind, target = {}) {
    if (submitting) return;
    const problems = yue2RequestProblems(kind, settings, target, problemContext);
    if (kind === "cover" && transcriptionReviewProblem) problems.push(transcriptionReviewProblem);
    if (problems.length) {
      setSubmitError(problems.join(" "));
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body = buildYue2JobRequest(kind, settings, target, requestedGpu);
      const response = await submitYue2Jobs(projectId, body, token);
      const created = Array.isArray(response?.jobs) ? response.jobs : [];
      setSubmitted((current) => [...created, ...current]);
    } catch (error) {
      setSubmitError(error);
      if (error?.code === "license_acknowledgment_required") {
        licenseLapsed();
      }
    } finally {
      setSubmitting(false);
    }
  }
  return { submitError, submitting, licenseLapsed, submit };
}
