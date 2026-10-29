# Independently configurable memory reasoners

## Problem and target

Boundary Write currently receives its reasoner host from the lifecycle hook, while Manage reads
the Store-wide executor. This couples the producing host to Write and prevents one Store from
choosing distinct reasoning transports for Write and Manage.

The target is an operation-specific policy layered over the existing executor configuration.
Explicit CLI choices win, then the operation policy, then legacy Store configuration. With no
new policy, automatic Write still uses the source host and Manage keeps using the legacy Store
executor. A selected reasoner failure is terminal for that attempt and never changes transport.

## Work units

1. Update the host-adapter contract and setup documentation for operation-specific selection,
   source identity, diagnostics, and compatibility.
2. Add configuration and routing tests for defaults, legacy TOML, operation policies, CLI
   precedence, endpoint combinations, fail-closed errors, and source-session provenance.
3. Implement nested executor policies, central resolution, source-only hook metadata, and
   channel-specific setup and Doctor checks without changing Write or Manage algorithms.
4. Run focused and complete automated verification, then perform attributable Codex/Muse live
   preflights and cross-host Store exercises where local authentication permits.
5. Commit, push, open a draft PR, inspect the diff for unrelated or sensitive content, and drive
   available CI checks green without merging.
