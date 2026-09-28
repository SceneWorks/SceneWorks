// SceneWorks Studio UI kit — root app. Wires the shell + screens, and drives
// data-theme / data-accent on <html> exactly like the real SceneWorks shell.
const { Icon, Button } = window.SceneWorksDesignSystem_b6febf;
const { Sidebar, Topbar } = window.SW_Shell;

function Placeholder({ title, body }) {
  return (
    <section className="page-frame">
      <div className="empty-panel">
        <span style={{ display: "grid", placeItems: "center", width: 52, height: 52, borderRadius: "var(--r-lg)", background: "var(--accent-soft)", color: "var(--accent-strong)" }}><Icon.Sparkle size={26} /></span>
        <div className="section-heading" style={{ justifyItems: "center" }}><h2>{title}</h2></div>
        <p className="view-copy" style={{ textAlign: "center" }}>{body}</p>
        <Button variant="primary" icon={<Icon.Plus size={16} />}>Get started</Button>
      </div>
    </section>
  );
}

const SCREENS = {
  Library: window.SW_Library,
  Image: window.SW_ImageStudio,
  Models: window.SW_Models,
  Video: () => <Placeholder title="Video Studio" body="Text- and image-to-video, clip extend, bridge, and person replacement — all rendered locally on your GPU." />,
  Character: () => <Placeholder title="Character Studio" body="Keep the same face across every shot using identity models and character LoRAs." />,
  Training: () => <Placeholder title="Training" body="Build captioned datasets and train image or video LoRAs locally — no cloud, no Python." />,
  Presets: () => <Placeholder title="Presets" body="Save and reuse recurring generation setups across studios." />,
  Queue: () => <Placeholder title="Queue" body="Running and recent jobs, with per-job routing and worker activity." />,
  Logs: () => <Placeholder title="Logs" body="This session's routing decisions and worker activity." />,
};

function App() {
  const [active, setActive] = React.useState("Image");
  const [theme, setTheme] = React.useState("light");
  const [accent, setAccent] = React.useState("teal");
  React.useEffect(() => {
    const html = document.documentElement;
    html.setAttribute("data-theme", theme);
    html.setAttribute("data-accent", accent);
  }, [theme, accent]);

  const Screen = SCREENS[active] ?? SCREENS.Image;
  return (
    <div className="app">
      <Sidebar active={active} setActive={setActive} />
      <div className="workspace">
        <Topbar active={active} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />
        <Screen />
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
