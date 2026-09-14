# SceneWorks release and hotfix workflow

This is the official process for maintaining a stable release line while larger
features continue on `main`. It covers SceneWorks hotfixes, inference changes,
version preparation, release publication, and failed-candidate recovery.

The process was exercised end to end for
[`v0.8.2`](https://github.com/SceneWorks/SceneWorks/releases/tag/v0.8.2) on
2026-08-07. The
[desktop release](https://github.com/SceneWorks/SceneWorks/actions/runs/31197369935)
and [RunPod publication](https://github.com/SceneWorks/SceneWorks/actions/runs/31197369437)
both completed successfully.

## Branch model

| Ref | Purpose | Mutation policy |
| --- | --- | --- |
| `main` | Ongoing development, including work not ready to release | Normal protected development branch |
| `release/next` | The tested code intended for the next release | Mutable through reviewed PRs only |
| `hotfix/gi-<issue>` | One SceneWorks hotfix based on `release/next` | Short-lived; delete after merge |
| `release/vX.Y.Z` | Temporary version-bump branch created by `scripts/release.mjs` | Short-lived; delete after merge |
| `release/X.Y.Z` | Exact source snapshot for a successful published release | Immutable after publication |
| `vX.Y.Z` | Annotated tag that triggers release automation | Immutable after publication |

`release/next` is the release source of truth. Do not tag `main` merely because
the legacy release script describes that older workflow. Do not merge hotfixes
into an existing `release/X.Y.Z` branch.

### Release invariants

1. `release/X.Y.Z` and the peeled annotated tag `vX.Y.Z^{}` point to the same
   commit.
2. The version bump and every intended hotfix or inference-pin update are merged
   into `release/next` before the immutable branch and tag are created.
3. A published release, its tag, and its versioned release branch are immutable.
4. A failed, unpublished candidate is provisional. Its draft, tag, and
   `release/X.Y.Z` branch may be deleted and recreated after fixing
   `release/next`.
5. Every release-only change is either already present on `main`, explicitly
   inapplicable to `main`, or forward-ported through a separate PR.
6. Use a clean checkout. If the normal checkout has unrelated changes, use an
   isolated clone or worktree rather than stashing, discarding, or mixing them
   into release work.

## Backport a SceneWorks hotfix

Start from the current release candidate, not from a versioned release branch:

```bash
git fetch origin
git switch -c hotfix/gi-<issue> origin/release/next
```

Identify the smallest complete fix on `main`. Inspect the source PR's commits
and dependencies; do not assume its GitHub merge commit is the desired unit.
Cherry-pick the necessary commits in dependency order:

```bash
git cherry-pick -x <source-commit>
```

Resolve conflicts in favor of the release line's architecture. Generated files
from a later `main` architecture are not automatically applicable: include them
only when their generator and source inputs exist on `release/next`, and
regenerate them from the release-line source whenever possible.

Validate the behavior that motivated the hotfix, then run the relevant package
checks. At minimum:

```bash
git diff --check origin/release/next...HEAD
npm run check
```

Add focused Rust, JavaScript, or integration tests appropriate to the change.
For a conflict-heavy backport, comparing stable patch IDs with the original
commit is a useful additional check; it does not replace behavioral tests.

Push the branch and open a PR against `release/next`:

```bash
git push -u origin hotfix/gi-<issue>
gh pr create --base release/next --head hotfix/gi-<issue>
```

Merge only after the required checks pass. Delete the hotfix branch afterward.
If the fix did not originate on `main`, open a forward-port PR or record why the
release-specific adaptation is not applicable there.

## Include an inference change

SceneWorks pins `SceneWorks/inference` by exact 40-character commit SHA. An
inference fix therefore lands in two repositories: the inference change first,
then a SceneWorks pin-update PR.

### 1. Prepare the inference release line

The inference repository follows the same mutable-candidate pattern. If its
`release/next` does not yet exist, bootstrap it once from the latest immutable
inference branch that SceneWorks currently pins:

```bash
git fetch origin
git switch -c release/next origin/release/<current-app-version>
git push -u origin release/next
```

Create the inference hotfix from inference `release/next`, validate it in that
repository, and merge its PR back into inference `release/next`. Do not point
SceneWorks at a hotfix branch.

### 2. Create the SceneWorks pin-update PR

Record the exact merged inference commit, then create a SceneWorks branch from
SceneWorks `release/next`:

```bash
git fetch origin
git switch -c hotfix/inference-<issue> origin/release/next
node scripts/bump-inference.mjs --sha <inference-sha40>
```

`bump-inference.mjs` is the authority for the files and checks tied to the pin.
Depending on the release line, it updates the Cargo manifests, `Cargo.lock`, and
provenance constants and may regenerate derived artifacts. Follow every
fail-closed remediation it prints. In particular, do not hand-edit a license,
provenance, compatibility, capability, or calibration digest to make a stale
audit appear current.

If the new inference pin moves the `mlx-rs` fork revision in `Cargo.lock`, the
macOS lanes' `Fetch prebuilt MLX` step looks for the fork's `prebuilt-<sha12>`
release for that revision (`scripts/fetch-prebuilt-mlx.sh`; the fork publishes
it on every push to its `main`). Until it exists the lanes build libmlx from
source once per cache miss and say so with a workflow annotation — check
`https://github.com/SceneWorks/mlx-rs/releases/tag/prebuilt-<sha12>` if the
macOS job is slower than usual after a bump (sc-21382).

Review the complete diff and run the repository checks plus relevant backend
tests. Push the branch and open a PR against SceneWorks `release/next`. The pin
PR must merge before the version bump is finalized.

Before tagging SceneWorks, create `release/X.Y.Z` in the inference repository at
the exact pinned inference SHA. That branch becomes immutable when the matching
SceneWorks release is successfully published. An inference tag is not required
by SceneWorks because Cargo consumes the exact SHA.

If there was no inference change, do not create a redundant pin PR. Verify that
the existing pinned SHA is still reachable through its immutable inference
release branch.

## Prepare the version

Do this only after all intended hotfix and inference-pin PRs have merged into
`release/next`.

Use a clean local branch named `release/next` that tracks
`origin/release/next`. The release script still assumes `main` when it creates a
PR, so use `--no-pr` and open the correctly based PR yourself:

```bash
git fetch origin
git switch release/next
git pull --ff-only origin release/next
npm run release -- X.Y.Z --any-branch --no-pr
git push -u origin release/vX.Y.Z
gh pr create \
  --base release/next \
  --head release/vX.Y.Z \
  --title "chore(release): X.Y.Z"
```

The script synchronizes the root npm version, desktop and web package versions
and lockfiles, Tauri version, Cargo workspace version, and `Cargo.lock`. Review
that complete version diff and merge the PR only after its checks pass.

Do not run `npm run release -- tag` for this workflow. Its tag phase implements
the older main-based process and does not create the required immutable release
branch.

## Cut the release

### 1. Preflight

Before creating either ref, verify all of the following:

- `release/next` contains the intended code and reports version `X.Y.Z`.
- No intended PR against `release/next` remains unmerged.
- The exact inference pin and any inference `release/X.Y.Z` branch are correct.
- `.github/workflows/release.yml` requires the `signing` runner label for the
  macOS job.
- The signing Mac is online. Only the Mac holding the Developer ID certificate
  and notary key may carry `signing`.
- No `release/X.Y.Z` branch, `vX.Y.Z` tag, or GitHub Release already exists.
- The checkout is clean and exactly matches `origin/release/next`.

### 2. Create the branch and annotated tag

Create the versioned branch first, then the tag. Pushing the tag triggers both
release workflows:

```bash
git fetch origin
git switch -c release/X.Y.Z origin/release/next
git rev-parse HEAD
git tag -a vX.Y.Z -m "SceneWorks vX.Y.Z" HEAD
git push -u origin release/X.Y.Z
git push origin vX.Y.Z
```

Immediately verify the invariant locally:

```bash
test "$(git rev-parse release/X.Y.Z)" = "$(git rev-parse 'vX.Y.Z^{}')"
```

The annotated tag object has its own SHA; compare the branch with the **peeled
tag commit**, not with the tag-object SHA.

### 3. Wait for both release workflows

The tag starts two independent workflows:

- `.github/workflows/release.yml` builds macOS first, creates the draft, then
  builds Windows and Linux and attaches their artifacts.
- `.github/workflows/publish-runpod.yml` builds and publishes
  `ghcr.io/sceneworks/sceneworks-runpod:X.Y.Z` and `:latest`.

The appearance of a draft after the macOS job is not completion. Wait for the
entire desktop workflow and the RunPod workflow to succeed.

### 4. Review and publish

Before pressing **Publish**, verify:

- the tag still peels to the `release/X.Y.Z` commit;
- macOS, Windows, Linux, and RunPod workflows all succeeded;
- the draft contains the DMG, macOS updater archive, Windows EXE and MSI, Linux
  AppImage and DEB, and the completed `latest.json`;
- release notes and prerelease status are correct; and
- the RunPod versioned image and `latest` were published by the successful run.

Publishing is the immutability boundary. After publication, never move or
delete the tag or versioned release branch and never replace the release's
artifacts. Corrections require a new patch version.

## Recover a failed, unpublished candidate

Use this recovery only when the release was never successfully published. If a
release was public or its artifacts may have been consumed, cut a new patch
version instead.

1. Cancel or wait for **every** workflow started by the failed tag, including
   RunPod. An old run must not publish or overwrite the same container tag after
   the replacement run starts.
2. Delete the unpublished draft without deleting unrelated tags.
3. Delete the failed tag and provisional versioned release branch.
4. Fix `release/next` through a reviewed PR.
5. Repeat the preflight and cut steps from the new `release/next` commit.

```bash
gh release delete vX.Y.Z --repo SceneWorks/SceneWorks --yes
git push origin :refs/tags/vX.Y.Z
git push origin --delete release/X.Y.Z
git tag -d vX.Y.Z  # only if the stale local tag exists
```

Do not merely rerun a workflow when its tagged commit contains the defect: a
rerun uses the same immutable workflow and source snapshot. Fix
`release/next`, recreate both refs at the corrected commit, and let the new tag
start fresh runs.

## Advance `release/next`

Continue merging selected fixes into `release/next` for patch releases. When a
larger feature on `main` is ready for release, promote it into `release/next`
through a reviewed PR and resolve any release-line differences deliberately.
Do not bypass `release/next` by tagging the feature branch or `main`.

After every release, confirm that all release-line fixes are accounted for on
`main`. The version-only release commit does not need to be merged back merely
to record that a tag exists; reconcile version metadata deliberately when the
next feature train is promoted.

## Success criteria

A release is complete only when:

- the versioned branch and peeled tag share one commit;
- the desktop and RunPod workflows succeeded from that commit;
- every expected desktop and updater artifact is present;
- the GitHub Release is published; and
- the versioned release refs are treated as immutable from that point forward.
