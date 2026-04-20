# Issue Labeling

Classify a GitHub issue for kUPS, a JAX-based molecular simulation library.

## Available GitHub Labels

{{LABELS}}

## Project Context

{{MANIFESTO}}

## Open Milestones

{{MILESTONES}}

## Issue

**Title:** {{ISSUE_TITLE}}

**Body:**
{{ISSUE_BODY}}

## Instructions

1. Pick exactly 3 labels from the list of available GitHub labels above: one type, one priority, one area
2. Use the EXACT label names as they appear in the list
3. If the issue relates to a milestone, set `milestone` to its exact title; otherwise `null`

## Output

Return ONLY valid JSON:
```json
{"labels": ["<exact label>", "<exact label>", "<exact label>"], "milestone": "<exact title>" or null, "reasoning": "<one sentence>" or null}
```
