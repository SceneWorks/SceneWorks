// Legacy UI tests exercise controls and transport without a running API. Give those
// tests a successful server response; mounted API and focused feedback tests own
// validation policy and typed refusal behavior.
globalThis.__sceneworksOptionsTestServer = async ({ requests }) => requests.map((resolved) => ({
    state: "valid", resolved, metadataSource: "server", liveChecks: "server",
}));
