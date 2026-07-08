# Security Policy

Thanks for helping keep SceneWorks and its users safe.

## Supported versions

SceneWorks is pre-1.0 and moves quickly. Security fixes land on `main` and in
the latest release. Please make sure you can reproduce an issue against the
**latest release or `main`** before reporting.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report privately using either of:

- **GitHub private vulnerability reporting** — go to the repository's
  **Security** tab → **Report a vulnerability** (preferred; keeps the report
  attached to the repo). This must be enabled by the maintainer under
  *Settings → Code security → Private vulnerability reporting*.
- **Email** — **michael@trefry.net**.

Please include as much of the following as you can:

- A description of the issue and its impact.
- Steps to reproduce, or a proof of concept.
- The affected version / commit, and the platform (macOS/MLX, Windows/candle,
  or Docker server).
- Any suggested remediation, if you have one.

## What to expect

- We'll acknowledge your report as soon as we can, and aim to give an initial
  assessment within a few days.
- We'll keep you updated on progress toward a fix and coordinate on disclosure
  timing.
- With your permission, we're happy to credit you when the fix is released.

Please give us a reasonable chance to release a fix before any public
disclosure.

## Scope notes

A few things are **documented, intentional tradeoffs** rather than
vulnerabilities — please read these before reporting:

- **Loopback trust on the desktop app.** When LAN remote access is enabled, any
  local process/user on the same machine can reach the loopback API without the
  access token. This is a deliberate single-user-desktop tradeoff, described
  under [Local access control](README.md#local-access-control). Running in
  remote-access mode on a shared, multi-user host is out of scope unless you've
  configured it as documented there.
- **Server credential storage.** On the Docker/server path, credentials are
  protected by `0600` file mode plus your orchestrator's secret handling, not
  app-level encryption. This is described in
  [Service credentials](README.md#service-credentials-api-tokens).
- **Model weights.** SceneWorks downloads third-party model weights at runtime.
  Issues in the weights themselves belong to their upstream projects, not here.

If you're unsure whether something is in scope, report it privately anyway and
we'll figure it out together.
