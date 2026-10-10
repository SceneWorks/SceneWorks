// SceneWorks Studio UI kit — Library screen (cosmetic recreation).
const { Icon, Button } = window.SceneWorksDesignSystem_b6febf;
const { AssetTile } = window.SW_Shell;

const TAGS = ["portrait", "product", "storyboard", "reference", "upscaled"];
const GRID = [
  { l: "keeper_close_v2", b: "★★★★" }, { l: "harbor_wide" }, { l: "lantern_detail", b: "upscaled" },
  { l: "storm_sky" }, { l: "cliffside_dawn" }, { l: "boat_moored" },
  { l: "rope_texture" }, { l: "window_light" }, { l: "keeper_hands" }, { l: "gull_flight" },
];

function LibraryScreen() {
  const [mode, setMode] = React.useState("assets");
  const [type, setType] = React.useState("all");
  return (
    <section className="page-frame">
      <div className="work-panel">
        <div className="work-panel-rule"></div>
        <div className="toolbar">
          <label className="file-upload-button"><Icon.Folder size={16} /> Import</label>
          <input type="search" placeholder="Search name, prompt, or tags…" />
          <select aria-label="Asset type" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="all">All media</option><option value="image">Images</option>
            <option value="video">Videos</option><option value="upload">Uploads</option>
          </select>
          <select aria-label="Asset tag" defaultValue="all">
            <option value="all">All tags</option>
            {TAGS.map((t) => <option key={t}>{t}</option>)}
          </select>
          <label className="checkline" style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-muted)", fontSize: 13 }}>
            <input type="checkbox" /> Rejected
          </label>
          <div className="segmented-control" role="group" aria-label="Asset collection">
            <button type="button" className={mode === "assets" ? "active" : ""} onClick={() => setMode("assets")}>Assets</button>
            <button type="button" className={mode === "trashcan" ? "active" : ""} onClick={() => setMode("trashcan")}>Trashcan</button>
          </div>
        </div>
        <div className="stat-strip">
          <div className="stat-chip"><span className="stat-chip-label">Project</span><span className="stat-chip-value">Portrait lookbook</span></div>
          <div className="stat-chip"><span className="stat-chip-label">Assets</span><span className="stat-chip-value">312 total</span></div>
          <div className="stat-chip"><span className="stat-chip-label">Images</span><span className="stat-chip-value">274</span></div>
          <div className="stat-chip"><span className="stat-chip-label">Clips</span><span className="stat-chip-value">38</span></div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 280px", gap: 18, alignItems: "start" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
          {GRID.map((g, i) => <AssetTile key={i} label={g.l} badge={g.b} />)}
        </div>
        <aside className="work-panel" style={{ gap: 14 }}>
          <div className="work-panel-rule"></div>
          <AssetTile label="keeper_close_v2" />
          <div className="section-heading"><h2 style={{ fontSize: 16 }}>keeper_close_v2</h2></div>
          <div style={{ display: "grid", gap: 6, fontSize: 12.5, color: "var(--text-muted)" }}>
            <div>Z-Image-Turbo · Q8 · 1024×1024</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11.5 }}>seed 84213 · cfg 4.5 · 28 steps</div>
          </div>
          <div className="rating-stars" style={{ alignSelf: "start" }}>
            {[0,1,2,3,4].map((i) => (
              <button key={i} type="button" className={i < 4 ? "star-rating-button active" : "star-rating-button"}><Icon.Star filled={i < 4} /></button>
            ))}
          </div>
          <div className="detail-actions">
            <Button variant="secondary" icon={<Icon.Image size={16} />}>Send to Image</Button>
            <Button variant="secondary" icon={<Icon.Editor size={16} />}>Edit</Button>
            <Button variant="danger" icon={<Icon.Logs size={16} />}>Discard</Button>
          </div>
        </aside>
      </div>
    </section>
  );
}

window.SW_Library = LibraryScreen;
