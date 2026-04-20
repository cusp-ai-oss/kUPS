# Weekly Summary

Generate a status report for stakeholders (PIs, collaborators, funders) on repository activity this week.

## Current Roadmap

{{ROADMAP}}

## This Week's Activity

{{ACTIVITY_DATA}}

## Instructions

This is for **non-technical stakeholders**. They care about:
- What was accomplished this week
- Progress toward roadmap objectives

They do NOT care about:
- Code implementation details
- Technical architecture decisions
- Specific file changes

Focus on **completed work** and how it relates to roadmap objectives.

## What to Report

1. **Merged PRs**: What capabilities were added/fixed and why they matter
2. **Closed Issues**: What problems were resolved
3. **New Issues**: What new work items were identified

## Output Format (follow the bullet point structure faithfully!)

```markdown
## Weekly Summary:

### Completed
[2-3 sentences: theme of the week's completed work and how it advances the roadmap]

**[Category]:** (group by roadmap objective or label)
- [What was done]
  - (#PR/#Issue)
  - [Why it matters for roadmap]

### New Issues
[2-3 sentences: theme of new issues raised this week and their significance]

**[Category]:** (group by roadmap objective or label)
- [Issue title]
  -(#number)
  - (urgency/priority)
  - (labels)
  - [one-line summary, why it matters/not matters]


*Contributors: @name1, @name2*
```

Keep it concise. Omit sections with no activity.
