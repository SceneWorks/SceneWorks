// SceneWorks Studio UI kit — Image Studio screen (cosmetic recreation).
const { Icon, Button } = window.SceneWorksDesignSystem_b6febf;
const { AssetTile } = window.SW_Shell;

const MODES = ["Text to image", "Edit", "With character"];
const MODELS = ["Z-Image-Turbo", "FLUX.2 [klein] 9B", "Qwen-Image", "Krea 2 (Turbo)", "SDXL / RealVisXL"];

function Field({ label, children }) {
  return <label className="settings-field" style={{ flex: "1 1 140px" }}>{label}{children}</label>;
}

function ImageStudio() {
  const [mode, setMode] = React.useState(0);
  const [prompt, setPrompt] = React.useState("cinematic portrait of a lighthouse keeper, 85mm, soft window light, film grain");
  return (
    <section className="page-frame">
      <div className="work-panel">
        <div className="work-panel-rule"></div>
        <div className="mode-tabs">
          {MODES.map((m, i) => (
            <button key={m} type="button" className={mode === i ? "mode-tab active" : "mode-tab"} onClick={() => setMode(i)}>
              {i === 0 ? <Icon.Image size={16} /> : i === 1 ? <Icon.Editor size={16} /> : <Icon.Character size={16} />}
              {m}
            </button>
          ))}
        </div>
        <div className="prompt-input-row">
          <textarea className="prompt-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Describe the image…" />
          <Button variant="primary" icon={<Icon.Sparkle />} style={{ alignSelf: "stretch", minWidth: 140 }}>Generate</Button>
        </div>
        <div className="settings-bar">
          <div className="settings-bar-row">
            <Field label="Model"><select defaultValue="Z-Image-Turbo">{MODELS.map((m) => <option key={m}>{m}</option>)}</select></Field>
            <Field label="Aspect"><select defaultValue="1:1"><option>1:1</option><option>3:2</option><option>16:9</option><option>2:3</option></select></Field>
            <Field label="Variations"><select defaultValue="4"><option>1</option><option>2</option><option>4</option><option>6</option></select></Field>
            <Field label="Quality"><select defaultValue="Q8"><option>bf16</option><option>Q8</option><option>Q4</option></select></Field>
          </div>
        </div>
      </div>

      <div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12 }}>
          <p className="eyebrow">Latest batch</p>
          <span style={{ fontSize: 12.5, color: "var(--text-muted)" }}>4 variations · seed 84213 · 28 steps</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
          <AssetTile badge="new" label="var 1" />
          <AssetTile label="var 2" />
          <AssetTile label="var 3" />
          <AssetTile label="var 4" />
        </div>
      </div>

      <div>
        <p className="eyebrow" style={{ marginBottom: 12 }}>Earlier today</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
          {["lighthouse dawn","harbor wide","keeper close","storm sky","lantern detail","cliffside"].map((l) => <AssetTile key={l} label={l} />)}
        </div>
      </div>
    </section>
  );
}

window.SW_ImageStudio = ImageStudio;
