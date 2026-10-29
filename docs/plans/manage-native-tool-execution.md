# Manage native-tool execution

## Preserved call chain

`mem sleep` and the one-shot Dream worker still enter `Manage.sleep()`. The method first performs
T0 maintenance deterministically: date normalization, weight settlement, co-occurrence links,
exact-duplicate invalidation, near-duplicate group repair, and clustering. It then computes T1
proposals, freezes their source hashes, and delegates only the review execution.

Text mode renders the original full prompt and parses JSON-line verdicts. Experimental native mode
sends compact proposal summaries to one isolated Codex or Muse Code run and serves a run-scoped
evidence/verdict toolkit over stdio MCP. Both return the same `ReviewOutcome`; Manage then enforces
per-kind caps and calls `decide()` only for a still-open proposal with unchanged sources.
`decide()` applies through Store methods and appends the Decision Ledger. Finally `Manage.sleep()`
writes the Dream Report and optionally commits the Store repository.

The manual path holds the Dream run and maintenance locks around `Manage.sleep()`. Automatic
Dreaming adds the persisted claim/lease and stable dream id, then the worker takes the same locks
and calls the same method. No resident backend is introduced.

## Reused transport and independent authority

PR #106's reusable piece is the executor transport: a Host can receive a per-job stdio MCP server.
The Write toolkit is not a Manage authority model. Manage does not expose Store record, correct,
delete, merge, generic memory, shell, or filesystem tools because those would bypass open-proposal
membership, caps, stale-target checks, the Decision Ledger, and report accounting.

`ManageReviewToolkit` exposes only:

- `list_proposals`: the authoritative menu for this run.
- `read_evidence`: bounded, pageable reads of targets named by one menu item.
- `submit_verdicts`: one complete judgement set under the existing verdict schema.

`Manage._review()` retains ownership of cap enforcement and calls the existing `decide()` path for
accepted verdicts. T0, proposal generation, claim/lease handling, locks, mutation guards, Decision
Ledger, Dream Report, and Git behavior remain unchanged. Text Host and endpoint paths remain the
default. The native tools validate run membership and source hashes, bound output and total
evidence, and persist a trace for the library to absorb after the Host exits.

## Runtime bounds and failure semantics

`manage.review_max_tool_calls`, per-result characters, and total evidence characters bound the MCP
surface; `[executor.manage].timeout_seconds` bounds the Host process. Codex's public CLI does not
offer a model-step flag, so `review_max_steps` is an audit ceiling used with the tool budget rather
than a claim that hidden model requests can be killed exactly. JSONL `turn.completed` supplies
token/cache totals and Host turn count when available. The executor timestamps JSONL lifecycle
events as they arrive: startup ends at the first event, MCP spans matched `mcp_tool_call`
start/completion events, shutdown begins at `turn.completed`, and the residual Host duration is
reported as model/runtime wait. Muse instead enforces its public eight-model-step flag and returns
JSONL through the existing Muse dialect; its current JSONL does not expose the Codex
`turn.completed` usage/phase contract, so token and phase fields remain unavailable rather than
being inferred. Manage separately times its existing verdict validation and apply loop. The four
Codex Host phases add to Host duration; they expose transport shape without retaining reasoning
text or granting a new capability.

A run with no accepted complete submission, a timeout, an invalid id, a stale source, or an
abnormal Host exit fails closed and never selects another reasoner. This limitation and the paired
quality/latency evidence keep native mode experimental.

## Muse Code reuse and lifecycle

Muse Native Manage adds no second adapter and no library-side ReAct implementation. It calls the
same `ToolHostReasoner`, `Host.converse`, `MuseCodeDialect`, isolated credential/routing staging,
stdio MCP injection, prompt-file transport, JSONL answer parser, process timeout, retry, and
OpenRouter namespace bridge already exercised by Muse Native Write. The only executor selection
change is that explicit native Manage now admits the canonical `muse-code` Host.

The per-run MCP server name is `memory_review`, so Muse receives only
`mcp__memory_review__list_proposals`, `mcp__memory_review__read_evidence`, and
`mcp__memory_review__submit_verdicts`. Personal settings, hooks, session logs, foreign context,
shell, web, and workspace writes stay out of the isolated run. The executor marker keeps any Host
lifecycle hook inert. Muse owns its Agent loop and MCP dispatch; the namespace bridge only flattens
the OpenRouter request schema so returned scalar names match Muse's dispatcher.

Muse Code 1.4.4 or newer is required for exact `run.toolset` selection. Authentication and provider
routing are copied from the existing Muse setup into the temporary profile; the Manage MCP server
is then the only staged server. A non-zero exit, timeout, missing complete submission, or rejected
tool call without a later valid complete submission fails closed. Unlike Codex, Muse has a public
`--max-model-steps` bound and no Codex analytics shutdown override; Codex alone supplies the
fine-grained phase and token counters currently reported by Manage.

## Authenticated Codex evidence

The opt-in system test completed the whole CLI path with local Codex 0.162.0 and
`gpt-5.6-sol` in 32.20 seconds: `mem sleep` generated a proposal, Codex called
`read_evidence`, then `submit_verdicts`, and Manage validated and wrote the Store, Decision
Ledger, and Dream Report. The test is skipped unless `AGENT_MEMORY_LIVE_CODEX=1` is set; the
normal suite never depends on credentials or a network service.

## Authenticated Muse evidence

Muse Code 1.4.9 with the existing OpenRouter/pproxy route completed two isolated Store runs on
2026-10-20. The opt-in test uses `AGENT_MEMORY_LIVE_MUSE_MANAGE=1`; normal CI remains credential
free.

| case | successful evidence reads | raw Session reads | verdict | Host wall | tokens |
|---|---:|---:|---|---:|---|
| supersede / multi-read | 2 | 0 | reject | 53.127s | unavailable |
| source ambiguity / raw enabled | 4 | 2 | reject | 72.136s | unavailable |

Both runs followed `Muse Native Agent -> Manage MCP -> submit_verdicts -> Manage validation ->
Decision Ledger / Dream Report`. The first trace included two rejected evidence calls and one
rejected incomplete submission before Muse corrected its arguments and submitted a complete
Verdict. The raw run opened both exact target-bound pointers,
`sessions/muse-confirmed#0-0` and `sessions/muse-guess#0-0`, then also corrected an invalid read and
an incomplete submission. These error/feedback/correction sequences demonstrate Muse's own Agent
loop and MCP dispatch, not a pre-scripted library loop. Both final Verdicts were rejections, so the
Store truth files remained active while Ledger and Report were written through the normal Manage
path. Muse's JSONL exposed no compatible token counters; the implementation reports them as
unavailable. This two-case execution validates integration, not quality or performance gains.

## Earlier Codex paired integration sample

`tools/manage_review_benchmark.py` defines a fixed 12-case corpus across merge, supersede,
split, abstract-review, and delete proposals. On 2026-10-19 one authenticated, paired run used
four byte-identical Store pairs and alternated arm order. Text agreed with the fixed labels on
9/12 cases; Native agreed on 10/12. Both arms had zero runtime errors and complete Ledger and
Report output. Native read every target, using 26 successful tool calls (4 list, 18 read, 4
submit). It reported 177,866 input, 137,728 cached-input, and 2,524 output tokens. The Text
compatibility invocation did not expose token usage; its rendered prompts totalled 10,171
characters.

Native took 110.766 seconds versus Text's 59.496 seconds, 86.2% slower in this sample. Its one
additional labelled agreement was an explicitly obsolete delete proposal; both arms rejected a
labelled complementary merge and current-policy supersede. With 12 cases and one replay this is
an integration sample, not an accuracy benchmark under the repository's experiment rules. The
trace contained no repeated evidence reads, so the run did not justify a speculative performance
commit. Native remains opt-in while a larger repeated corpus and Host portability are open work.

## Follow-up phase diagnostic

On 2026-10-19 a bounded follow-up replay selected b1 and b2 only: six cases, two paired batches,
four Codex invocations. Native completed all six evidence checks and all 13 tool calls without an
error. Its 69.593 seconds of Host duration split into 3.115 seconds startup, 53.622 seconds
model/runtime wait, 0.050 seconds MCP calls, and 12.806 seconds shutdown; Manage apply added 0.137
seconds. MCP plus apply were under 0.3% of Native wall time.

Text completed too, but its b2 call took 232.049 seconds while b1 took 17.927 seconds, so this
single replay does not support an arm-level speed comparison. It does isolate the Native cost
shape: measured MCP execution and Manage apply are not the bottleneck. Because every requested
target was read exactly once and MCP time was negligible, the evidence did not license even a
small speculative preload. The implementation therefore keeps the original ReAct loop, summary-
first prompt, on-demand evidence reads, and permission boundary unchanged.

### Codex shutdown tail

A 2026-10-20 process-level trace replayed three representative Native Manage cases without
shortening the Agent Turn. All three completed their evidence reads, submitted a complete verdict
set, and wrote the full Ledger and Report. After the last MCP response, Codex spent 1.19-1.45
seconds producing its final agent message; `turn.completed` followed 1-11 milliseconds later. The
MCP service and its transport process were gone about 6 milliseconds after `turn.completed`.

The remaining 5.49-7.04 seconds occurred inside the Codex process after MCP shutdown. Native
process exit was followed by stdout/stderr EOF and the parent `wait()` within about 4-5
milliseconds, excluding repeated waits and pipe-buffer backpressure. Codex 0.162.0 performs a
synchronous analytics flush after its processor and MCP shutdown. A same-case diagnostic with
only `analytics.enabled=false` reduced shutdown from 5.51 seconds to 0.041 seconds while
preserving the complete turn and fail-closed result handling. Native Manage now disables that
per-run host analytics flush; Native Write, ReAct behavior, MCP tools, and Store permissions are
unchanged. Three production-path replays then completed merge, split, and delete reviews with
shutdown times of 0.082, 0.056, and 0.044 seconds respectively.
