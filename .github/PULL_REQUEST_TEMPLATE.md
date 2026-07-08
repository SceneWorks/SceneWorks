<!--
Thanks for contributing to SceneWorks! Please fill this out so reviewers have
the context they need. See CONTRIBUTING.md for the full process.
-->

## What does this PR do?

<!-- A clear description of the change and why it's needed. -->

## Related issue

<!-- e.g. "Closes #123". For anything non-trivial, please open an issue first. -->

Closes #

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation only
- [ ] Refactor / chore (no user-facing behavior change)

## How was this tested?

<!-- Describe how you verified the change. Which platform(s) did you test on? -->

- [ ] macOS (Apple Silicon / MLX)
- [ ] Windows (NVIDIA / candle)
- [ ] Docker server (candle / CPU)
- [ ] Not platform-specific (web UI, docs, shared crate)

## Checklist

- [ ] I ran `npm run rust:check` (if I touched Rust) and it passed
- [ ] I ran `npm --prefix apps/web run lint && npm --prefix apps/web run test && npm --prefix apps/web run build` (if I touched the web app)
- [ ] I ran `npm run check` (scaffold checks)
- [ ] I updated documentation where relevant
- [ ] All my commits are signed off (`git commit -s` — see the DCO section in CONTRIBUTING.md)
- [ ] My PR is focused on a single logical change
