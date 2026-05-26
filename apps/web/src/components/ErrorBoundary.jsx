import React from "react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("SceneWorks render failed", error, info);
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    const message = this.state.error?.message ?? "Unknown render error";
    return (
      <main className="app-fallback" role="alert">
        <section className="empty-panel app-fallback-panel">
          <p className="eyebrow">SceneWorks</p>
          <h1>Something went wrong</h1>
          <p className="view-copy">{message}</p>
          <div className="toolbar">
            <button className="secondary-action" onClick={() => this.setState({ error: null })} type="button">
              Try again
            </button>
            <button className="primary-action" onClick={() => window.location.reload()} type="button">
              Reload
            </button>
          </div>
        </section>
      </main>
    );
  }
}
