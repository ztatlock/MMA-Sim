# AGENTS.md

Rules for AI agents working in this repo.

## Absolute rules

- **Never commit without explicit user review and approval.** Stage and show
  the diff; wait for the user to say "commit". This applies to every commit,
  every time — no exceptions, no "obvious" cases, no amending.
- **Never push without explicit user approval.** Same rule, same reason.
- **Never run destructive git operations** (`reset --hard`, `push --force`,
  branch deletion, `clean -f`) without explicit approval.

## Working conventions

- Plan lives in [ROADMAP.md](ROADMAP.md). Campaign notes go in [logs/](logs/).
- The Python impl under `mmasim/` is the bit-exact oracle. Do not modify it
  without a clear reason; rewrites land alongside it, not in place of it.
- Every performance claim must cite a benchmark. Every correctness claim must
  cite a corpus diff.
