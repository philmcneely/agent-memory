# Native tool use for Write negotiation

## Problem and target

Write negotiation runs a library-owned loop over a text-in, text-out reasoner. Looks and final
operations are JSON lines the library parses, every round re-renders the whole transcript into
one prompt, and a round cap, a forced last round and a format retry bound the exchange. On host
reasoners each round is also a cold one-turn CLI process whose own agent loop, tools and system
prompt are switched off. Host CLIs are themselves tool-using agents; the library wraps a second,
hand-written agent loop around a crippled one.

The session-reuse benchmark in PR #105 (Codex, fixed three-round scenario) locates the cost.
CLI resume cut client prompt characters by two thirds yet model input tokens by under a tenth,
and ran slower: the model receives the full history every turn whoever holds it. The app-server
arm's wall-time and token reductions came from one process per batch and from replacing the
host's coding-agent prompt and tools.

The target is the industry-standard tool-use agent, which lets the model decide what to look at,
how often, and when it is done:

- **One toolkit** in the core: recall, read, and operation submission.
- **Endpoint reasoner**: a standard loop — structured message history, native function calling,
  the loop ends when the model stops calling tools.
- **Host reasoner (`claude-code`, `codex`)**: the host runs its own loop. The same toolkit is
  exposed through a per-batch stdio MCP adapter and the distillation instructions replace the
  host's system prompt.

## Decisions

1. **No look budget, no turn cap, no forced last round, no format retry.** The model decides how
   many tools to call and when to stop. Only the existing per-request and per-process timeouts
   remain, as operational failure detection rather than reasoning limits.
2. **One toolkit, two transports.** The toolkit owns handle extension from tool results,
   active-only reads, and submission validation. The endpoint loop calls it in process; the MCP
   adapter serves it to the host. Neither transport carries tool semantics (Invariant 8). The
   core still holds no model client (Invariant 5).
3. **Final operations are a tool call.** `submit_operations` takes the batch's operations under a
   schema defined once in the toolkit. A submission is checked against the reconcile sheet and
   then by the Store's own write preparation (a check that persists nothing), so every refusal
   returns to the model inside the same run; the latest accepted submission is the batch's
   answer. A run that ends without an accepted submission writes nothing and leaves the backlog in
   place. Only sibling collisions or concurrent changes reach the existing repair path; the single
   write path is unchanged (Invariant 2). Added after the first paired run showed Haiku
   submissions passing the sheet check and then failing in the Store.
4. **State crosses the process boundary through the batch scratch directory.** The library seeds
   the toolkit (Store root and the batch's reconcile sheet) into a private per-batch directory;
   the MCP adapter opens the Store read-only and records looks, extended handles and the accepted
   submission there. The library absorbs that record after the host exits.
5. **Host posture is tools-only.** Claude Code: strict MCP config, no built-in tools, only the
   toolkit's tools allowed, system prompt replaced. Codex: user config and rules ignored,
   ephemeral, read-only sandbox, base instructions replaced, the MCP server required and
   pre-approved, local execution features off. Claude's bare mode is not used because it drops
   OAuth and would break zero-key install. The executor marker keeps hooks inert.
6. **The prompt splits into instructions and material.** Native forms send the distillation
   instructions and type table as the system prompt and the sheet plus conversation as the user
   message; the output contract lives in the tool schema. Constant material first keeps prefixes
   cacheable across batches.
7. **Other hosts keep the text protocol.** `hermes` and `muse-code` stay on the current
   negotiation, now calling the same toolkit for recall and read, until their MCP posture passes
   a live preflight. Manage review and Write repair remain single-shot text calls.
8. **Supersedes the session-reuse transports in PR #105.** Process and context reuse fall out of
   the host owning its loop.

## Work units

1. Add the Write negotiation design doc; update the host-adapter contract and the design index.
2. Unit tests: toolkit relationships (handle extension, unseen handles refused, active-only
   reads, submission refusal then acceptance, latest submission wins, seed and record round
   trip); endpoint loop against a fake chat endpoint (multiple and parallel tool calls, natural
   stop, no submission, malformed arguments, unknown tools); MCP adapter framing and catalogue;
   Claude Code and Codex tool command shapes.
3. Red-team and system tests: injection through recalled bodies, unseen handles, illegal
   operations, malformed arguments, an adapter that never writes the Store; an offline distill
   through fake `claude` and `codex` binaries that start the configured MCP server and drive a
   scripted look-then-submit sequence.
4. Implement the toolkit, split prompt, endpoint loop, MCP adapter, host postures and routing.
5. Verify: full suite, lint and types; live Codex and Claude Code distills on a fixture Store
   with recorded CLI versions; a paired comparison against the text protocol under identical
   Recall (Invariant 9), artefacts in the main working tree under `experiments/`.
6. Sweep design docs, README and `docs/TODO.md`; commit, push, open the PR and drive CI green.
