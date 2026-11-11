# Automated review

Review one kUPS pull request. CI has passed, so skip anything ruff, pyrefly
or `ci/kups_lint.py` enforce.

- Rules: `CLAUDE.md` at the workspace root (from main, authoritative).
- Code: the PR head is in `pr-head/`. Read it, don't run it.
- Context: `gh pr view $PR`, `gh pr diff $PR`, and `facts.md`: changed lines
  no test runs, new definitions nothing references and, on a re-review, how
  many lines changed since the last review. These are measured; trust them.
- Scope: `$LENS` and `$FILES`. On a re-review, only lines changed since the
  last review are in scope; everything else was reviewed already.

Only these findings count. Anything else (style, naming, wording, docs that
are imprecise but not false, refactors, stronger tests for code that a test
already runs, hypotheticals, and the PR title or description, which guide
the author) is not reported. Having no findings is a
completely valid result; don't look for something to say.

## `gate`

Work these in order; report the first that fails with `"kind": "gate"` and
stop. Judge from the description, the file list and a skim of the code.

1. Reviewable: one concern? If `$FILES` says the PR is over the size limit,
   this gate fails: name the split lines (by concern or dependency order).
2. Understandable: can you tell what problem this solves? If not, ask the
   specific questions.
3. Needed: is the problem real and not already handled elsewhere?
4. Best approach: does it fix the root cause, extend an existing interface
   rather than add a parallel one, and use what kUPS or a dependency already
   has? Search before saying it doesn't exist; name the better approach.

Also report, with `"kind": "duplicate"`, new code that reimplements a named
kUPS or dependency function, and, with `"kind": "dead"`, entries of
`facts.md` "New definitions nothing references" that are not public API
the PR description promises.

## `correctness`

Go through each changed function in `$FILES` and ask one question: is there
an input this PR's code gets wrong (wrong value, unit, dtype, shape, index or
gradient, an error where a result is due, a silent result where an error is
due, a retrace, or blow-up in time or memory)? Try the edges: zero, empty,
one element, NaN, overflow, integer or mixed inputs, several systems, every
periodicity. Report it with `"kind": "bug"` only if you can name the input
and trace it through the code to the wrong outcome.

Then go through `facts.md` "Changed lines no test runs" for your files and
report, with `"kind": "untested"`, each entry that is behaviour (a branch,
an error path, a formula), not trivial plumbing such as a `__repr__` or a
re-export. One finding per function.

## Escalate only when a human must look

Set `escalate` to one sentence if you can't settle something and a wrong
call is costly: physics or numerics you can't verify, changed simulation
results, public API or file-format breaks, or CI, workflow or secrets
changes.

## Output

Don't post anything: a second pass verifies your findings and posts the
review. Write `findings.json`:

```json
{
  "escalate": null,
  "body": [{"kind": "gate", "text": "point not tied to one line"}],
  "comments": [{"kind": "bug", "path": "src/...", "line": 42, "side": "RIGHT", "body": "..."}]
}
```

Inline comments go on diff lines. Each point is one sentence, plus the
failing input for a bug. Never restate the PR or report checks that passed.
