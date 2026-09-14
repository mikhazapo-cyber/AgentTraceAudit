# Recurring patterns

Reference only. The detector does not import this file. Checks are derived per trace from the task, the instructions, and the declared tools (`src/traceaudit/derive.py`). These are patterns seen in manual review, recorded so a human can interpret findings.

## Redundant re-execution

The agent repeats a call whose earlier identical invocation already returned a usable result, sometimes narrating a "correction" that changes nothing. No new information, and reusing the earlier result was already available. Look-alike: a retry after a transient failure (429, timeout) where the second result differs, or a new argument after feedback.

## Phantom-failure narration

The agent says a step failed or was malformed while the captured result shows success. Often paired with a redundant repeat. Family `evidence_contradiction`.

## Undeclared or hollow tool calls

Calls to names outside the declared inventory, or required analytical fields passed as empty strings or arrays so the tool runs but returns nothing. Look-alike: a harness wrapper such as `env_action` that *contains* a logical tool name from a text protocol.

## Argument-class mistakes

Passing a type or category where the protocol requires an entity or variable. The environment usually answers "cannot be executed". Look-alike: a speculative `read` of a missing file (ENOENT) that the agent abandons.

## Instruction-gated tools

Instructions restrict *when* a well-formed call is allowed, for example "only if the question asks for a superlative". The call can succeed and still be a violation.

## Skipped policy preconditions

A policy requires a check (authenticate, verify a balance) before a consequential action. The action can still succeed, which hides the omission from outcome-based scoring.

## Unverified success claims

A final message claims a world effect (emailed, refunded, deleted) that no tool result supports. Characterisations such as "here is a comprehensive plan" are not this family.

## Same approach after feedback

The agent repeats the same arguments after a validation error instead of changing the request. Look-alike: changing the target after an error and then succeeding.

## Capture artifacts, not agent errors

- Recorder `loss` events describe missing capture, not behaviour.
- `run.outcome: completed` is capture metadata, not task success.
- Duplicated native records (`model.response` and `assistant_message`) are one decision after the normalizer deduplicates by call id.
- A missing tool result may be an unpaired call (`capture_gaps`). Abstain instead of inventing a finding.
