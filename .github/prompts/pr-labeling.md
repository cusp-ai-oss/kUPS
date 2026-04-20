# PR Labeling

Auto-label a pull request based on changed files.

## PR Details

**Title:** {{PR_TITLE}}

**Changed Files:**
{{CHANGED_FILES}}

## Available Labels

{{AVAILABLE_LABELS}}

## Existing Labels on this PR

{{EXISTING_LABELS}}

## Instructions

Select appropriate `area:` labels based on the changed files:
- `area: core` — changes to core library code (src/kups/, main modules)
- `area: features` — feature implementations, new capabilities
- `area: api` — API changes, interfaces, public methods
- `area: infra` — CI/CD, workflows, build system, config
- `area: performance` — optimizations, benchmarks
- `area: docs` — documentation, examples, tutorials

Only suggest labels from the available labels list. Skip labels already applied.

## Output

Return valid JSON:
```json
{"labels": ["area: infra"] or []}
```
