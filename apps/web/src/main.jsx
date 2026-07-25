import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import { ErrorBoundary } from "./components/ErrorBoundary.jsx";
import { recordStartupMark } from "./startupTiming.js";
import "./styles.css";

recordStartupMark("bootstrap-start");

const rootElement = typeof document === "undefined" ? null : document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <ErrorBoundary>
      <App />
    </ErrorBoundary>,
  );
}

export { App } from "./App.jsx";
export { ErrorBoundary } from "./components/ErrorBoundary.jsx";
export { eventUrl } from "./api.js";
