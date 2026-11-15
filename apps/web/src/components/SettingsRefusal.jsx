const UPGRADE_CODES = new Set([
  "InvalidSchemaVersion", "UnsupportedSchemaVersion", "ConflictingAliases",
  "CarrierTooDeep", "MissingRecipeVersion", "InvalidCarrier",
]);

export function isSettingsRefusal(error) {
  return typeof error?.field === "string" && UPGRADE_CODES.has(error?.code);
}

// The failed subtree stays unmounted, so its autosave effects cannot overwrite the original.
export function SettingsRefusal({ error }) {
  return (
    <section className="page-frame empty-panel" role="alert" data-refusal-code={error.code}>
      <h2>Saved settings could not be opened</h2>
      <p>{error.message}</p>
      <p>Field: {error.field}</p>
      <p>Your saved settings have been kept. Open them with a compatible version of SceneWorks.</p>
      <button type="button" onClick={() => window.location.reload()}>Reload</button>
    </section>
  );
}
