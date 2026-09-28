import React, { useState } from "react";

const pendingScreenImports = new Set();

export async function waitForLazyScreenImports() {
  while (pendingScreenImports.size) {
    await Promise.allSettled([...pendingScreenImports]);
    await Promise.resolve();
  }
}

class LazyLoadErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    if (this.props.interaction) {
      return (
        <div className="lazy-interaction-boundary" role="alert">
          <span>{this.props.label} could not be loaded.</span>
          <button onClick={this.props.onRetry} type="button">
            Try again
          </button>
        </div>
      );
    }

    return (
      <section className="page-frame lazy-screen-boundary" role="alert">
        <h2>{this.props.label} could not be loaded</h2>
        <p>The screen bundle did not finish loading. Check the connection and try again.</p>
        <button onClick={this.props.onRetry} type="button">
          Retry
        </button>
      </section>
    );
  }
}

function ScreenLoading({ label }) {
  return (
    <section
      aria-busy="true"
      aria-live="polite"
      className="page-frame lazy-screen-boundary"
      role="status"
    >
      Loading {label}…
    </section>
  );
}

function InteractionLoading({ label }) {
  return (
    <span aria-busy="true" aria-live="polite" className="lazy-interaction-boundary" role="status">
      Loading {label}…
    </span>
  );
}

function createLazyNamedComponent(importComponent, exportName) {
  return React.lazy(async () => {
    const request = importComponent();
    pendingScreenImports.add(request);
    try {
      const module = await request;
      const component = module[exportName];
      if (!component) {
        throw new Error(`Lazy module does not export ${exportName}`);
      }
      return { default: component };
    } finally {
      pendingScreenImports.delete(request);
    }
  });
}

/**
 * Build a retryable React.lazy boundary around a named screen export.
 *
 * Keeping lazy component creation inside this wrapper lets Retry invoke the
 * import function again after a transient chunk-load failure. The wrapper
 * remains mounted inside App's keep-alive panes, so successful screens retain
 * their existing local state across navigation.
 */
export function lazyScreen(importScreen, exportName, label) {
  const createLazyComponent = () => createLazyNamedComponent(importScreen, exportName);

  function LazyScreen(props) {
    const [{ attempt, Screen }, setLoadState] = useState(() => ({
      attempt: 0,
      Screen: createLazyComponent(),
    }));
    const retry = () =>
      setLoadState((current) => ({
        attempt: current.attempt + 1,
        Screen: createLazyComponent(),
      }));

    return (
      <LazyLoadErrorBoundary
        key={attempt}
        label={label}
        onRetry={retry}
      >
        <React.Suspense fallback={<ScreenLoading label={label} />}>
          <Screen {...props} />
        </React.Suspense>
      </LazyLoadErrorBoundary>
    );
  }

  LazyScreen.displayName = `Lazy${exportName}`;
  return LazyScreen;
}

/** Build a compact retryable lazy boundary for controls that appear only after an interaction. */
export function lazyInteraction(importComponent, exportName, label) {
  const createLazyComponent = () => createLazyNamedComponent(importComponent, exportName);

  function LazyInteraction(props) {
    const [{ attempt, Component }, setLoadState] = useState(() => ({
      attempt: 0,
      Component: createLazyComponent(),
    }));
    const retry = () =>
      setLoadState((current) => ({
        attempt: current.attempt + 1,
        Component: createLazyComponent(),
      }));

    return (
      <LazyLoadErrorBoundary
        interaction
        key={attempt}
        label={label}
        onRetry={retry}
      >
        <React.Suspense fallback={<InteractionLoading label={label} />}>
          <Component {...props} />
        </React.Suspense>
      </LazyLoadErrorBoundary>
    );
  }

  LazyInteraction.displayName = `Lazy${exportName}`;
  return LazyInteraction;
}
