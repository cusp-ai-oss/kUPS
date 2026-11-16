# Verify and post a review

`findings.json` holds draft findings on PR $PR from one or more reviewers
you can't ask. Check each one yourself before anything is posted.

- Rules: `CLAUDE.md`. Code: `pr-head/` (read, don't run). Context:
  `gh pr view $PR`, `gh pr diff $PR`, and `facts.md` (measured; trust it).

Keep a finding only if its `kind` is one of these and its check holds:

- `gate`: the PR is over the size limit in `facts.md`, or you agree it
  mixes concerns, its problem is unclear or not real, or a concrete better
  approach exists.
- `bug`: trace the named input through the code yourself and reach the
  same wrong outcome. No input, or a trace you can't confirm: drop.
- `untested`: the lines are listed in `facts.md` and are behaviour, not
  plumbing.
- `duplicate`: the named existing function really does the same thing.
- `dead`: the definition is listed in `facts.md` and the PR description
  doesn't promise it as public API.

Drop everything else, any finding about the PR title or description, and
any finding that questions a choice the PR description or a docstring
explains, or that is about code outside the
re-review scope in `facts.md`. Merge findings that make the same point, add
nothing of your own, and keep wording unless a fact in it is wrong. Keep
`escalate` only if its reason holds. Having nothing left is a completely
valid result (the review is then `LGTM`); when in doubt, drop.

If a `gate` finding is kept, post only gate findings and end the body with
"Detail review after this is settled.". Strip the `kind` fields before
posting.

## Post

Post one review with event `COMMENT`: write it to `review.json` and send it
with `gh api repos/$REPO/pulls/$PR/reviews --input review.json`, with
`commit_id` set to `$SHA`.

- Inline comments are the kept `comments`.
- The body is the `text` of the kept `body` points, one per line, cut to
  what no inline comment says. If `escalate` is kept, start the body with
  `**Needs human review:** <its sentence>`. If nothing at all is kept, the
  body is `LGTM`.
- Every body ends with `<!-- kups-review:$SHA base:$BASE -->`.
