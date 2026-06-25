---
name: commit-message
description: Generate a Conventional Commits message from the current staged/unstaged diff, show it for approval, then create the git commit. Use when user says "commit this", "write a commit message", "/commit-message", or asks to commit current changes.
---

# Commit message skill

Generate technical, diff-grounded commit messages. Format: [Conventional Commits](https://www.conventionalcommits.org/).

## Workflow

1. Run `git status` and `git diff` (staged + unstaged) to see what changed. If nothing staged, diff working tree.
2. Run `git log -5 --oneline` to match repo's existing tone/prefix style.
3. Read the **full diff**, not just diffstat — pull concrete function/class/variable names changed, not vague summaries.
4. Draft subject + optional body, show to user for approval before committing.
5. On approval: `git add` relevant files (specific paths, never `-A`/`.`), then `git commit` with the approved message via heredoc.

## Subject line

`<type>(<optional scope>): <imperative summary>`, ≤ 50 chars where possible, no period.

| type     | when |
|----------|------|
| feat     | new user-visible capability |
| fix      | bug fix |
| refactor | code restructure, no behavior change |
| perf     | performance improvement |
| docs     | docs/comments only |
| chore    | deps, tooling, config, no source logic change |
| test     | test-only changes |

Match this repo's existing convention: `type: summary` (no parens scope used historically — check `git log` before adding one).

## Body (only when "why" isn't obvious from the diff)

- Explain motivation/root cause, not what the diff already shows.
- Name the specific function/constant/mechanism touched if it disambiguates intent (e.g. race condition guard, threshold constant).
- Wrap at ~72 cols. Blank line between subject and body.
- Skip body entirely for small, self-evident changes.

## Rules

- Never invent rationale not visible in diff or user's stated intent — ask if unclear.
- Squash multiple files into one logical commit message only if they're one coherent change; otherwise flag to user that diff looks like multiple unrelated changes and ask whether to split.
- Never commit files likely containing secrets (`.env`, `credentials.json`, etc.) — warn user instead.
- Never use `--no-verify` or `--amend` unless user explicitly asks.
- Always confirm message with user before running `git commit`.
