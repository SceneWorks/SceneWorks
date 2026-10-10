// SceneWorks Studio UI kit — Model Manager screen (cosmetic recreation).
const { Icon, Button } = window.SceneWorksDesignSystem_b6febf;

const TABS = ["Image", "Video", "Utility", "LoRAs"];
const MODELS = {
  Image: [
    { name: "Z-Image / Z-Image-Edit", note: "Turbo · edit-capable", size: "6.6 GB", mem: "12 GB", state: "installed" },
    { name: "FLUX.2 [klein] 9B", note: "Non-commercial license", size: "17.2 GB", mem: "24 GB", state: "download" },
    { name: "Qwen-Image (+ Edit)", note: "Text render · edit", size: "12.1 GB", mem: "18 GB", state: "installed" },
    { name: "Krea 2 (Raw / Turbo)", note: "Photoreal", size: "9.4 GB", mem: "16 GB", state: "download" },
    { name: "SDXL / RealVisXL", note: "SDXL-family kernel", size: "6.9 GB", mem: "10 GB", state: "installed" },
    { name: "InstantID", note: "Identity model", size: "2.1 GB", mem: "8 GB", state: "gated" },
  ],
  Video: [
    { name: "LTX-2.3", note: "Text- & image-to-video", size: "14.8 GB", mem: "24 GB", state: "installed" },
    { name: "Wan 2.2 (TI2V-5B)", note: "Single-expert", size: "10.2 GB", mem: "18 GB", state: "download" },
    { name: "Stable Video Diffusion", note: "Image-to-video", size: "9.6 GB", mem: "16 GB", state: "download" },
  ],
  Utility: [
    { name: "Real-ESRGAN", note: "Upscaler", size: "0.7 GB", mem: "4 GB", state: "installed" },
    { name: "JoyCaption", note: "Vision captioner", size: "8.1 GB", mem: "12 GB", state: "download" },
    { name: "Anubis-8B", note: "Prompt refiner", size: "8.0 GB", mem: "12 GB", state: "download" },
  ],
  LoRAs: [
    { name: "lighthouse-look", note: "Image LoRA · Z-Image", size: "0.2 GB", mem: "—", state: "installed" },
    { name: "product-studio", note: "Image LoRA · SDXL", size: "0.3 GB", mem: "—", state: "installed" },
  ],
};

function StateCell({ state }) {
  if (state === "installed") return <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--success)", fontSize: 12.5, fontWeight: 600 }}><span className="dot"></span> Installed</span>;
  if (state === "gated") return <Button variant="secondary" icon={<Icon.Info size={15} />}>Add token</Button>;
  return <Button variant="primary" icon={<Icon.Sparkle size={15} />} style={{ minHeight: 36, padding: "0 14px", fontSize: 13 }}>Download</Button>;
}

function ModelRow({ m }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 100px 100px 150px", alignItems: "center", gap: 12, padding: "12px 14px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--r-md)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
        <span style={{ display: "grid", placeItems: "center", width: 38, height: 38, borderRadius: "var(--r-sm)", background: "var(--accent-soft)", color: "var(--accent-strong)", flex: "0 0 auto" }}><Icon.Model size={20} /></span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{m.note}</div>
        </div>
      </div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, color: "var(--text-muted)" }}>{m.size}</div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, color: "var(--text-muted)" }}>{m.mem}</div>
      <div style={{ justifySelf: "end" }}><StateCell state={m.state} /></div>
    </div>
  );
}

function ModelManager() {
  const [tab, setTab] = React.useState("Image");
  return (
    <section className="page-frame">
      <div className="work-panel">
        <div className="work-panel-rule"></div>
        <div className="work-panel-head">
          <div className="work-panel-head-text">
            <p className="eyebrow work-panel-eyebrow">Local catalog</p>
            <div className="section-heading"><h2>Models</h2></div>
            <p class="work-panel-hint">Weights download on first use — nothing is bundled. Each model lists size and minimum memory.</p>
          </div>
          <div className="work-panel-actions"><Button variant="secondary" icon={<Icon.Folder size={16} />}>Import checkpoint</Button></div>
        </div>
        <div className="toolbar">
          <div className="segmented-control">
            {TABS.map((t) => <button key={t} type="button" className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}
          </div>
          <input type="search" placeholder="Search models…" />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 100px 100px 150px", gap: 12, padding: "0 14px" }}>
        <span className="stat-chip-label">Model</span>
        <span className="stat-chip-label">Size</span>
        <span className="stat-chip-label">Min memory</span>
        <span></span>
      </div>
      <div style={{ display: "grid", gap: 8 }}>
        {MODELS[tab].map((m) => <ModelRow key={m.name} m={m} />)}
      </div>
    </section>
  );
}

window.SW_Models = ModelManager;
