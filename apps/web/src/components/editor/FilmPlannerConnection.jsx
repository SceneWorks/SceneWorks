import React, { useEffect, useMemo, useState } from "react";

import {
  listFilmPlannerConnections,
  saveFilmPlannerConnection,
  testFilmPlannerConnection,
} from "../../api/filmPlannerConnections.js";
import {
  hasPresentCredential,
  loadCredentials,
  normalizeCredentialHost,
  saveCredential,
} from "../../credentials.js";

function emptyForm() {
  return {
    id: "",
    label: "",
    baseUrl: "",
    credentialHost: "",
    supportsModelListing: true,
    supportsImageInput: false,
    timeoutSeconds: 60,
    maxOutputTokens: 8192,
  };
}

function connectionId(label) {
  const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
  const suffix = globalThis.crypto?.randomUUID?.().replaceAll("-", "").slice(0, 10)
    ?? Math.random().toString(36).slice(2, 12);
  return `connection-${slug || "openai"}-${suffix}`;
}

export default function FilmPlannerConnection({ planning, disabled, active, token, onChange, onNotice }) {
  const [connections, setConnections] = useState([]);
  const [credentials, setCredentials] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [secret, setSecret] = useState("");
  const [models, setModels] = useState([]);
  const [working, setWorking] = useState(false);

  const selected = useMemo(
    () => connections.find((connection) => connection.id === planning.connectionId),
    [connections, planning.connectionId],
  );

  useEffect(() => {
    let canceled = false;
    Promise.all([listFilmPlannerConnections(token), loadCredentials()])
      .then(([saved, storedCredentials]) => {
        if (canceled) return;
        setConnections(saved ?? []);
        setCredentials(storedCredentials ?? []);
      })
      .catch((error) => !canceled && onNotice(error.message));
    return () => { canceled = true; };
  }, [token, onNotice]);

  useEffect(() => {
    if (selected) setForm({ ...selected });
  }, [selected]);

  function selectConnection(id) {
    const connection = connections.find((item) => item.id === id);
    setModels([]);
    onChange((next) => {
      next.planning.connectionId = id || undefined;
      if (!id) next.planning.modelId = undefined;
      next.planning.sendReferencePixels = Boolean(
        connection?.supportsImageInput && next.planning.sendReferencePixels,
      );
    });
    setForm(connection ? { ...connection } : emptyForm());
  }

  async function saveConnection() {
    setWorking(true);
    try {
      const id = form.id || connectionId(form.label);
      const baseHost = normalizeCredentialHost(form.baseUrl);
      const savedHost = normalizeCredentialHost(form.credentialHost);
      const host = secret.trim() || savedHost === baseHost ? baseHost : "";
      if (secret.trim()) {
        setCredentials(await saveCredential({
          host,
          label: `Film planner: ${form.label.trim()}`,
          scheme: "bearer",
          token: secret,
        }));
      }
      const saved = await saveFilmPlannerConnection(id, {
        label: form.label,
        baseUrl: form.baseUrl,
        credentialHost: host || undefined,
        supportsModelListing: form.supportsModelListing,
        supportsImageInput: form.supportsImageInput,
        timeoutSeconds: Number(form.timeoutSeconds),
        maxOutputTokens: Number(form.maxOutputTokens),
      }, token);
      setConnections((items) => [saved, ...items.filter((item) => item.id !== saved.id)]);
      setForm({ ...saved });
      setSecret("");
      onChange((next) => {
        next.planning.connectionId = saved.id;
        if (!saved.supportsImageInput) next.planning.sendReferencePixels = false;
      });
      onNotice("Planning connection saved. Its credential remains in the host secret facility.");
    } catch (error) {
      onNotice(error.message);
    } finally {
      setWorking(false);
    }
  }

  async function testConnection() {
    if (!planning.connectionId) return;
    setWorking(true);
    try {
      const result = await testFilmPlannerConnection(planning.connectionId, token);
      setModels(result.models ?? []);
      onNotice(result.detail);
    } catch (error) {
      onNotice(error.message);
    } finally {
      setWorking(false);
    }
  }

  const locked = disabled || active || working;
  const hasCredential = selected?.credentialHost
    ? hasPresentCredential(credentials, selected.credentialHost)
    : false;
  return (
    <section aria-labelledby="film-planner-connection-heading" className="ve-film-connection">
      <h4 id="film-planner-connection-heading">OpenAI-compatible connection</h4>
      <div className="ve-film-grid">
        <label>Saved connection
          <select aria-label="Saved planning connection" disabled={locked} value={planning.connectionId ?? ""} onChange={(event) => selectConnection(event.target.value)}>
            <option value="">Create or choose a connection</option>
            {connections.map((connection) => <option key={connection.id} value={connection.id}>{connection.label}</option>)}
          </select>
        </label>
        <label>Connection label<input aria-label="Planning connection label" disabled={locked} value={form.label} onChange={(event) => setForm((value) => ({ ...value, label: event.target.value }))} /></label>
        <label>Base URL<input aria-label="Planning connection base URL" disabled={locked} placeholder="https://provider.example/v1" value={form.baseUrl} onChange={(event) => setForm((value) => ({ ...value, baseUrl: event.target.value }))} /></label>
        <label>Credential<input aria-label="Planning connection credential" autoComplete="new-password" disabled={locked} placeholder={hasCredential ? "Saved credential (leave blank to keep)" : "Optional bearer token"} type="password" value={secret} onChange={(event) => setSecret(event.target.value)} /></label>
        <label>Timeout (seconds)<input aria-label="Planning connection timeout" disabled={locked} max="300" min="5" type="number" value={form.timeoutSeconds} onChange={(event) => setForm((value) => ({ ...value, timeoutSeconds: event.target.value }))} /></label>
        <label>Maximum output tokens<input aria-label="Planning maximum output tokens" disabled={locked} max="65536" min="256" type="number" value={form.maxOutputTokens} onChange={(event) => setForm((value) => ({ ...value, maxOutputTokens: event.target.value }))} /></label>
      </div>
      <label><input checked={form.supportsModelListing} disabled={locked} onChange={(event) => setForm((value) => ({ ...value, supportsModelListing: event.target.checked }))} type="checkbox" /> Endpoint supports model listing</label>
      <label><input checked={form.supportsImageInput} disabled={locked} onChange={(event) => setForm((value) => ({ ...value, supportsImageInput: event.target.checked }))} type="checkbox" /> Endpoint supports image input</label>
      <div className="ve-film-actions">
        <button disabled={locked || !form.label.trim() || !form.baseUrl.trim()} onClick={saveConnection} type="button">Save connection</button>
        <button disabled={locked || !planning.connectionId} onClick={testConnection} type="button">Test and list models</button>
      </div>
      <div className="ve-film-grid">
        {models.length ? (
          <label>Listed model
            <select aria-label="Listed planner model" disabled={locked} value={models.includes(planning.modelId) ? planning.modelId : ""} onChange={(event) => onChange((next) => { next.planning.modelId = event.target.value || undefined; })}>
              <option value="">Choose a listed model</option>
              {models.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
        ) : null}
        <label>Planner model ID<input aria-label="External planner model ID" disabled={locked} placeholder="Model ID (manual entry is always available)" value={planning.modelId ?? ""} onChange={(event) => onChange((next) => { next.planning.modelId = event.target.value || undefined; })} /></label>
      </div>
      <div className="ve-film-disclosure">
        <strong>Data sent when you start external planning</strong>
        <p>Destination: {selected?.baseUrl || form.baseUrl || "Choose a connection"}</p>
        <p>SceneWorks sends the script, edited brief, beats and dialogue, reference role names and descriptions, and the target video model capability envelope. A saved credential is sent to this selected endpoint only in the Authorization header; SceneWorks excludes it from the planner prompt, project files, logs, and exports. Local file paths and other project assets are not sent. Reference image bytes remain local unless you enable the option below.</p>
        <label><input checked={Boolean(planning.sendReferencePixels)} disabled={locked || !selected?.supportsImageInput} onChange={(event) => onChange((next) => { next.planning.sendReferencePixels = event.target.checked; })} type="checkbox" /> Send approved reference image pixels to this image-capable endpoint</label>
        {!selected?.supportsImageInput ? <p>Reference pixels remain local because this connection is not marked image-capable.</p> : null}
      </div>
    </section>
  );
}
