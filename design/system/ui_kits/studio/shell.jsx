// SceneWorks Studio UI kit — app shell (sidebar + topbar). Cosmetic recreation of
// the real SceneWorks web shell, composing design-system primitives + classes.
const { Logo, Wordmark, Icon, StatusDot, AccentPicker, CompactSelector } = window.SceneWorksDesignSystem_b6febf;

const NAV_STUDIOS = [
  { id: "Library", label: "Library", icon: Icon.Library },
  { id: "Image", label: "Image Studio", icon: Icon.Image },
  { id: "Video", label: "Video Studio", icon: Icon.Video },
  { id: "Character", label: "Character Studio", icon: Icon.Character },
  { id: "Training", label: "Training", icon: Icon.Train },
];
const NAV_MANAGE = [
  { id: "Models", label: "Model Manager", icon: Icon.Model, badge: "64" },
  { id: "Presets", label: "Presets", icon: Icon.Preset },
  { id: "Queue", label: "Queue", icon: Icon.Queue },
  { id: "Logs", label: "Logs", icon: Icon.Logs },
];

const PROJECTS = [
  { id: "look", name: "Portrait lookbook" },
  { id: "promo", name: "Product promo" },
  { id: "story", name: "Storyboard — Act I" },
];

function NavItem({ item, active, onClick }) {
  const IconCmp = item.icon;
  return (
    <button type="button" className={active ? "nav-item active" : "nav-item"} onClick={onClick}>
      <IconCmp />
      <span className="nav-label">{item.label}</span>
      {item.badge ? <span className="nav-badge">{item.badge}</span> : null}
    </button>
  );
}

function Sidebar({ active, setActive }) {
  const [project, setProject] = React.useState("look");
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark"><Logo size={30} /></span>
        <h1>Scene<span className="light">Works</span></h1>
      </div>
      <div className="sidebar-section">
        <CompactSelector
          items={PROJECTS}
          selectedId={project}
          onSelect={(p) => setProject(p.id)}
          onCreate={() => {}}
          createLabel="New project"
          getSubtitle={() => "Local project"}
          label="Project"
        />
      </div>
      <div className="sidebar-section">
        <div className="sidebar-section-title">Studios</div>
        <div className="nav-list">
          {NAV_STUDIOS.map((item) => (
            <NavItem key={item.id} item={item} active={active === item.id} onClick={() => setActive(item.id)} />
          ))}
        </div>
      </div>
      <div className="sidebar-section">
        <div className="sidebar-section-title">Manage</div>
        <div className="nav-list">
          {NAV_MANAGE.map((item) => (
            <NavItem key={item.id} item={item} active={active === item.id} onClick={() => setActive(item.id)} />
          ))}
        </div>
      </div>
      <div className="sidebar-footer">
        <span className="app-version">SceneWorks 0.7.5 · MLX</span>
      </div>
    </aside>
  );
}

const TITLES = {
  Library: { title: "Library", hint: "Every still and clip across this project" },
  Image: { title: "Image Studio", hint: "Text-to-image, edits, and reference-guided generation" },
  Video: { title: "Video Studio", hint: "Text- and image-to-video, extend, and person replace" },
  Character: { title: "Character Studio", hint: "Keep one face across every shot" },
  Training: { title: "Training", hint: "Caption datasets and train LoRAs locally" },
  Models: { title: "Model Manager", hint: "Download and manage local checkpoints" },
  Presets: { title: "Presets", hint: "Saved generation setups" },
  Queue: { title: "Queue", hint: "Running and recent jobs" },
  Logs: { title: "Logs", hint: "This session's routing and worker activity" },
};

function Topbar({ active, theme, setTheme, accent, setAccent }) {
  const meta = TITLES[active] ?? TITLES.Image;
  return (
    <header className="topbar">
      <div className="topbar-title">
        <h1>{meta.title}</h1>
        <p>{meta.hint}</p>
      </div>
      <div className="topbar-spacer"></div>
      <span className="topbar-status">
        <span className="status-pill"><StatusDot ok /> worker online</span>
      </span>
      <button type="button" className="queue-chip"><Icon.Queue size={15} /> 2 running</button>
      <span className="topbar-divider"></span>
      <button type="button" className="icon-btn" aria-label="Notifications"><Icon.Bell /></button>
      <AccentPicker accent={accent} onChange={setAccent} />
      <button type="button" className="icon-btn" aria-label="Toggle theme" onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
        {theme === "light" ? <Icon.Moon /> : <Icon.Sun />}
      </button>
    </header>
  );
}

// Checkerboard placeholder tile matching SceneWorks' own thumb aesthetic.
function AssetTile({ ratio = "1 / 1", label, badge }) {
  return (
    <div style={{ position: "relative", aspectRatio: ratio, borderRadius: "var(--r-md)", overflow: "hidden", border: "1px solid var(--border)",
      background: "repeating-linear-gradient(45deg, color-mix(in oklch, var(--accent) 20%, transparent) 0 6px, transparent 6px 12px), color-mix(in oklch, var(--warm) 14%, var(--surface))" }}>
      {badge ? <span style={{ position: "absolute", top: 8, left: 8, padding: "2px 8px", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--accent-fg)", background: "var(--accent)", borderRadius: "var(--r-pill)" }}>{badge}</span> : null}
      {label ? <span style={{ position: "absolute", bottom: 8, left: 8, right: 8, fontSize: 11, fontWeight: 500, color: "var(--text)", background: "color-mix(in oklch, var(--surface) 82%, transparent)", padding: "3px 7px", borderRadius: "var(--r-xs)", backdropFilter: "blur(4px)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span> : null}
    </div>
  );
}

window.SW_Shell = { Sidebar, Topbar, AssetTile };
