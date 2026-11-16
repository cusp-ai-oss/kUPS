# kUPS

Read `docs/notebooks/conventions.md` before changing library code. Run
`uv run pre-commit run --all-files` before pushing. It covers ruff,
pyrefly, `ci/kups_lint.py` and the lock file. Fix findings rather than
silencing them, and use `# noqa: CODE` only where a rule is wrong for that
line.

## Code
- Group related arrays in `Table`s so cardinality (per system, per
  particle) is in the type. Update state with lenses.
- Use the `Index` API (`valid_mask`, `indices_in`, `sum_over`, `Table[index]`,
  ...) rather than `raw_indices`, which is only for numeric kernels.
- Compose existing parts (`SequentialPropagator`, `StateProperty`,
  `CachePropagator`, `Cell` methods, `segment_*`) before writing a class.
  Put behaviour on the type that owns the data.
- Search for a sibling implementation before adding an algorithm, and
  unify rather than duplicate.
- Choose a default instead of adding a flag. Use one code path and one
  public entry point. Use `lax.cond`, not flags stored in the state.
- Derived values are properties, not fields. No factories, no thin
  wrappers, no type aliases that only rename.
- Use the narrowest correct type, `@overload` over duplicated functions, and
  typed config instead of string keys. Avoid `Any` in public return types,
  nested ones included.
- State holds data, not functions. Inject behaviour as functions instead of
  enums, strategies, factories or `abc` hierarchies.
- Types carry the contract: no `getattr`/`hasattr` probing, `Any`, or
  runtime type checks where a protocol, overload or `Table` type can say it.
  Runtime checks on values use kUPS assertions.
- Data protocols are named `Is*`, `Has*` or `Supports*`. Reuse the existing
  vocabulary (keys and queries, `_params`, `_lens`) instead of new synonyms.
- Plain code over clever code: no bit tricks or walrus chains where a loop
  or a named type reads better.

## Numerics
- Take tolerances from `jnp.finfo`.
- Out-of-bounds indices equal the axis size, never `-1`.
- Performance changes keep behavior, values and gradients 1:1, with a test.
- If code can't run under `jit`, write it in NumPy.
- Watch for recompiles: keep caches in the state, not on static fields of
  immutable dataclasses, and keep shapes stable across calls.

## Writing
Applies to docstrings, docs, commit messages, PR descriptions and review
comments.
- Fewer words win whenever they carry the same information.
- Short and plain, in full sentences, the way a colleague would say it.
  No filler ("This PR addresses…", "comprehensive", "robust"), no padded
  bullet lists, no AI or tool attribution.
- Code and docs describe what the code does now. History and measurements
  go in the commit message or PR description.
- Docs pages start with a runnable example, then the details.

### Commits and PRs
- Titles are `type(scope): …` and state the resulting behaviour
  ("fix: indices_in keeps every out-of-range id out of range"), not the
  activity ("fix bug in indices_in").
- The description scales with the PR: one or two sentences of what and why
  for a small change. For larger ones also say what changes for users,
  which tests pin it, when the new path does *not* apply, and the stack
  position ("Part 3/5").
- Say whether results change. If values move, say by how much and why.

### Performance PRs
- Always include a table: one row (or column) per benchmark config, before
  and after (or Δ%), with the metric and unit (e.g. MC steps/s).
- State the hardware, precision and method, e.g. "one CPU core, median of 3
  interleaved runs".
- In a stack, give the change vs the previous PR and the whole stack vs
  `main`. Report compile time when it changes.

### Review comments and replies
- A comment is one sentence, often a question. Add a second only for the
  concrete input that shows a bug.
- A reply says what changed and names the test that pins it ("Added
  `test_x`; fails without the fix."), or says in one sentence why not,
  with a number if it's about performance.

## Tests and scope
- Every behaviour change gets a test that fails without it. Cover all
  periodicities, x32/x64 and multiple systems, using the smallest system
  that shows the behaviour.
- A bug fix comes with a minimal test that fails before the fix and passes
  after it.
- One concern per PR, with no unrelated files. In a stack, fold fixes into
  the PR that introduced the bug.
- Names follow the domain and the paper (TMMC, not "NVTw").
