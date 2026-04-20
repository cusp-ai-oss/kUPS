# Milestone Sync

Sync GitHub milestones with ROADMAP.md objectives.

## Roadmap

{{ROADMAP}}

## Existing Milestones

{{EXISTING_MILESTONES}}

## Recent Changes (git diff)

{{ROADMAP_DIFF}}

## Instructions

1. Parse ONLY quarters that exist in the roadmap (respect the diff - removed content means removed quarters/objectives)
2. Match objectives to existing milestones semantically (same work = match, even if renamed)
3. Categorize each:
   - `create`: New objective with no matching milestone
   - `update`: Milestone exists but title/description changed
   - `keep`: Milestone matches objective exactly
   - `stale`: Milestone has no matching objective (if diff shows removal, mark as stale!)
4. Calculate due_date for each quarter as the last day of that quarter (YYYY-MM-DD format)

## Output

Return ONLY valid JSON. Include ALL quarters from the roadmap:
```json
{
  "quarters": [
    {
      "quarter": "Q1 2026",
      "due_date": "2026-03-31",
      "objectives": [
        {"title": "Core Stability", "description": "..."},
        {"title": "Performance", "description": "..."}
      ]
    },
    {
      "quarter": "Q2 2026",
      "due_date": "2026-06-30",
      "objectives": [...]
    }
  ],
  "actions": {
    "create": [{"quarter": "Q1 2026", "title": "Core Stability", "description": "..."}],
    "update": [{"milestone_number": 1, "new_title": "New Name", "new_description": "..."}],
    "keep": [{"milestone_number": 2, "title": "Existing"}],
    "stale": [{"milestone_number": 3, "reason": "Not in any current quarter"}]
  }
}
```

IMPORTANT:
- Titles should NOT include the quarter prefix (e.g., "Core Stability" not "Q1 2026: Core Stability")
- The quarter prefix will be added by the script
