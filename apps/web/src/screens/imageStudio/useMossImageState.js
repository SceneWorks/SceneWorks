import { useConsentRequestOwner } from "../../hooks/useConsentRequestOwner.js";
import { mossAdvancedDefaults } from "../../mossImageAdvanced.js";
import { isMossImage, isMossEdit, mossEditDefaults, mossDefaults, mossAccelerated, mossComponentState } from "../../mossImage.js";
import { useEffect, useRef, useState } from "react";

// Extracted from ImageStudio along its existing concern boundary.
export function useMossImageState({
  model,
  saved,
  mode,
  imageModels,
  activeProject,
  token,
  jobs,
  downloadConditionalComponents
}) {
  const [mossImage, setMossImage] = useState(() => saved.mossImage ?? mossDefaults());

  const [mossImageEdit, setMossImageEdit] = useState(() => saved.mossImageEdit ?? mossEditDefaults());

  const [mossImageAdvanced, setMossImageAdvanced] = useState(() => saved.mossImageAdvanced ?? mossAdvancedDefaults());

  const [mossPromptBatch, setMossPromptBatch] = useState(() => saved.mossPromptBatch ?? []);

  const [mossDmdSteps, setMossDmdSteps] = useState(saved.mossDmdSteps ?? 8);

  const [mossDownload, setMossDownload] = useState({ busy: false, error: "", jobs: [] });

  const moss = isMossImage(model);

  const mossEdit = isMossEdit(model, mode);

  const mossDmd = moss && !mossEdit && mossAccelerated(mossImage);

  const mossComponents = moss ? mossComponentState(imageModels.find((item) => item.id === model), mossImage, mode) : { ready: true };

  const mossDownloadScope = `${activeProject?.id ?? ""}/${model}/${mode}/${mossComponents.purpose ?? ""}`;

  const captureMossDownloadOwner = useConsentRequestOwner(token, imageModels, undefined, mossDownloadScope);

  useEffect(() => { setMossDownload({ busy: false, error: "", jobs: [] }); }, [token, mossDownloadScope]);

  useEffect(() => { setMossDownload((current) => current.busy ? { ...current, busy: false } : current); }, [imageModels]);

  const mossDownloadJobs = mossDownload.jobs.map((queued) => jobs.find((job) => job.id === queued.id) ?? queued);

  const mossDownloadProgress = mossDownloadJobs.map((job) => `${job.payload?.files?.join(", ") ?? "component"}: ${job.status ?? "queued"}${Number.isFinite(job.progress) ? ` ${Math.round(job.progress * 100)}%` : ""}`).join("; ");

  const setMossField = (key, value) => setMossImage((current) => ({ ...current, [key]: value }));

  async function downloadMossAcceleration() {
    const ownsRequest = captureMossDownloadOwner(token);
    if (!ownsRequest()) return;
    setMossDownload({ busy: true, error: "", jobs: [] });
    try {
      const result = await downloadConditionalComponents(model, mossComponents.purpose);
      if (!ownsRequest()) return;
      setMossDownload({ busy: false, error: "", jobs: result?.jobs ?? [] });
    } catch (error) {
      if (!ownsRequest()) return;
      setMossDownload({ busy: false, error: error.message, jobs: [] });
    }
  }

  const mossEditTransition = useRef({ active: mossEdit, ordinary: null, replay: false });
  return { mossImage, setMossImage, mossImageEdit, setMossImageEdit, mossImageAdvanced, setMossImageAdvanced, mossPromptBatch, setMossPromptBatch, mossDmdSteps, setMossDmdSteps, mossDownload, moss, mossEdit, mossDmd, mossComponents, mossDownloadProgress, setMossField, downloadMossAcceleration, mossEditTransition };
}
