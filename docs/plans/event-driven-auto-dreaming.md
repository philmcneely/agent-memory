# Event-driven automatic Dreaming

## Baseline

The shared `main` baseline already provides lifecycle capture, detached boundary Distillation,
independent Write and Manage reasoner policies, Host reasoners, Manage, Store locking, atomic
truth writes, optimistic correction checks, Dream reports, and the decision ledger. It has no
automatic Manage trigger. Fresh Host setup also leaves the manual Manage default on Claude Code
after Codex or Muse setup.

This implementation adds no server, listener, scheduler thread, or daemon.

## Decisions

1. Count distinct, fully distilled session identities from persisted watermarks. Pending Write
   work blocks Manage.
2. Reuse Manage's first-run and subsequent-run due semantics.
3. Claim only after successful boundary Distillation; use SessionStart and duplicate boundaries
   as retry or elapsed-time ticks.
4. Persist claim, lease, attempt, Host, session snapshot, outcome, report, and failure detail in
   Store state.
5. Fence duplicate workers with claim and run locks. Fence Distillation against Manage with a
   shared/exclusive maintenance lock and serialize duplicate distillers per session.
6. Treat the stable automatic Dream report as the completion record. Recover it before retrying
   a stale claim.
7. Use the claiming lifecycle Host for automatic Manage. On a fresh Host-backed Store, seed the
   otherwise empty manual Manage policy from the first successful Host setup.
8. Keep manual sleep and expose automatic status through the CLI.
9. Offer an opt-in, Store-scoped user crontab entry that only wakes the same due-aware CLI path;
   never install it from setup or add a Memory-owned scheduler.

## Verification

- Unit: due rules, distinct sessions, pending Write delay, Host affinity, launch shape, failure,
  lease recovery, report recovery, manual baseline, duplicate Hook behavior, and setup defaults.
- Cross-process: simultaneous Host/cron claims and duplicate workers.
- Cron: idempotent Store-scoped install, status, update, and removal while preserving unrelated
  crontab entries; explicit WSL scheduler diagnostics.
- Regression: full unit, system, red-team, lint, and type checks.
- Live: isolated HOME, Store, project, and logs; real Codex lifecycle Write/Read; automatic Manage
  with a real model decision; process exit and no listening port; delayed elapsed-time eligibility
  caught by the next event.

## Recovery boundary

A caught worker error is failed immediately. An uncatchable exit remains running until its lease;
the next event reclaims it after confirming that no worker owns the run lock. A report written
before a state-update crash proves completion. A crash before that report may leave reversible,
individually atomic Manage changes; retry reconciles current Store truth and never bypasses Store
validation or optimistic checks. A full persistent job queue and supervisor are not provided.
