---
name: changelog
description: Log project updates into CHANGELOG.md in concise bullet-point format following Keep a Changelog conventions. Use when user says "log this", "update changelog", "/changelog", or after committing user-visible changes.
---

# Changelog skill

Maintain `CHANGELOG.md` at repo root. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## File structure

```markdown
# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- ...
### Changed
- ...
### Fixed
- ...
### Removed
- ...

## [v0.1.0] - YYYY-MM-DD
...
```

Only include section headers (`### Added` etc.) that have at least one bullet.

## Source of truth

Diff against last logged state: `git log` since last changelog update, or current staged/unstaged diff if asked to log "this change" directly. Read commit messages + diffstat, not full diffs, to determine intent.

## Section mapping (commit type → section)

| commit prefix / intent       | section |
|-------------------------------|---------|
| feat                          | Added   |
| fix                           | Fixed   |
| refactor / perf / chore       | Changed — only if user-visible behavior changes |
| remove / deprecate            | Removed |
| breaking change                | Added/Changed, prefixed `**BREAKING:**` |

Skip entirely: pure deps/lockfile bumps, formatting/lint-only commits, merge commits, WIP commits, internal refactors with no user-visible effect.

## Bullet rules

- One bullet = one logical change. Squash multiple related commits into a single bullet.
- Present tense, imperative: "Add wake word detection", not "Added wake word detection".
- State what changed + impact. Add why only if non-obvious (hidden constraint, bug root cause, breaking rationale).
- No issue/PR numbers, no file paths — those belong in git history, not changelog.
- Flag breaking changes explicitly with `**BREAKING:**` prefix.

## Placement

- New entries go under `## [Unreleased]`, newest section at top if multiple types changed.
- Create `## [Unreleased]` header if missing.
- On release/tag, rename `[Unreleased]` to `[vX.Y.Z] - YYYY-MM-DD` and start a fresh empty `[Unreleased]` above it.
