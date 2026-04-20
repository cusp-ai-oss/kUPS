/**
 * @fileoverview Sync GitHub milestones from ROADMAP.md using LLM
 * @description Uses Claude to parse roadmap objectives and semantically match
 * them to existing milestones. Handles creating new milestones, updating renamed
 * objectives, and marking stale milestones.
 * @see .github/workflows/sync-milestones.yml
 */

const fs = require('fs');

/** @type {{model: string, max_tokens: number}} */
const MODEL_CONFIG = JSON.parse(fs.readFileSync('.github/config/models.json', 'utf8')).milestone_sync;

const { execSync } = require('child_process');

/**
 * Gets the git diff for ROADMAP.md from the last commit
 * @returns {string} The diff output or empty string if no diff
 */
function getRoadmapDiff() {
  try {
    return execSync('git diff HEAD~1 HEAD -- ROADMAP.md', { encoding: 'utf8' });
  } catch {
    return '';
  }
}

/**
 * Calls Claude to parse roadmap and determine milestone actions
 * @param {string} roadmapContent - Full contents of ROADMAP.md
 * @param {Array<Object>} existingMilestones - Current open milestones from GitHub
 * @param {string} roadmapDiff - Git diff showing recent changes
 * @param {string} apiKey - Anthropic API key
 * @returns {Promise<Object>} Actions to perform: { quarter, actions: { create, update, keep, stale } }
 * @throws {Error} If API returns an error response or invalid JSON
 */
async function syncMilestonesWithLLM(roadmapContent, existingMilestones, roadmapDiff, apiKey) {
  const prompt = fs.readFileSync('.github/prompts/milestone-sync.md', 'utf8')
    .replace('{{ROADMAP}}', roadmapContent)
    .replace('{{EXISTING_MILESTONES}}', JSON.stringify(existingMilestones, null, 2))
    .replace('{{ROADMAP_DIFF}}', roadmapDiff || '(No recent changes)');

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      model: MODEL_CONFIG.model,
      max_tokens: MODEL_CONFIG.max_tokens,
      messages: [{ role: 'user', content: prompt }]
    })
  });

  const result = await response.json();
  if (result.error) {
    throw new Error(`API error: ${result.error.message}`);
  }

  const text = result.content[0].text.trim();
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  return JSON.parse(jsonMatch ? jsonMatch[0] : text);
}

/**
 * Main entry point for GitHub Actions
 * @param {Object} params - GitHub Actions context
 * @param {Object} params.github - Octokit REST client
 * @param {Object} params.context - GitHub Actions context with repo info
 * @param {Object} params.core - GitHub Actions core utilities (unused but available)
 */
module.exports = async ({ github, context, core }) => {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  // Read roadmap
  let roadmapContent;
  try {
    roadmapContent = fs.readFileSync('ROADMAP.md', 'utf8');
  } catch (e) {
    console.log('No ROADMAP.md found, skipping milestone sync');
    return;
  }

  console.log('Read ROADMAP.md, sending to LLM for parsing and matching...');

  // Fetch existing milestones (open ones)
  const { data: allMilestones } = await github.rest.issues.listMilestones({
    owner,
    repo,
    state: 'open',
    per_page: 100
  });

  const existingMilestones = allMilestones.map(m => ({
    number: m.number,
    title: m.title,
    description: m.description || '',
    open_issues: m.open_issues,
    closed_issues: m.closed_issues
  }));

  console.log(`Found ${existingMilestones.length} existing open milestones`);

  // Get roadmap diff to help LLM understand what changed
  const roadmapDiff = getRoadmapDiff();
  if (roadmapDiff) {
    console.log('Roadmap diff detected:\n', roadmapDiff.slice(0, 500));
  }

  // Use LLM to parse roadmap and determine actions
  let result;
  try {
    result = await syncMilestonesWithLLM(roadmapContent, existingMilestones, roadmapDiff, process.env.ANTHROPIC_API_KEY);
  } catch (e) {
    console.error('LLM sync failed:', e.message);
    return;
  }

  const { quarters = [], actions } = result;

  // Build a map of quarter -> due_date for easy lookup
  const quarterDueDates = {};
  for (const q of quarters) {
    const dueDateISO = q.due_date ? `${q.due_date}T00:00:00Z` : null;
    quarterDueDates[q.quarter] = dueDateISO;
    console.log(`Parsed quarter: ${q.quarter}, due date: ${dueDateISO}, objectives: ${q.objectives?.length || 0}`);
  }

  console.log('LLM determined actions:', JSON.stringify(actions, null, 2));

  // Helper to strip quarter prefix if accidentally included
  const stripQuarterPrefix = (title) => {
    return title.replace(/^Q[1-4]\s+\d{4}:\s*/i, '');
  };

  // Execute actions

  // 1. Create new milestones
  for (const obj of (actions.create || [])) {
    const cleanTitle = stripQuarterPrefix(obj.title);
    const quarter = obj.quarter || quarters[0]?.quarter || 'Q1 2026';
    const title = `${quarter}: ${cleanTitle}`;
    const dueDateISO = quarterDueDates[quarter] || null;

    console.log(`Creating milestone: ${title}`);

    await github.rest.issues.createMilestone({
      owner,
      repo,
      title,
      description: obj.description || `Roadmap objective: ${cleanTitle}`,
      due_on: dueDateISO
    });
  }

  // 2. Update/rename existing milestones
  for (const update of (actions.update || [])) {
    const milestone = existingMilestones.find(m => m.number === update.milestone_number);
    if (!milestone) continue;

    // Preserve the quarter from the existing milestone title, or use the update's quarter
    const existingQuarter = milestone.title.match(/^(Q[1-4]\s+\d{4}):/)?.[1];
    const quarter = update.quarter || existingQuarter || quarters[0]?.quarter || 'Q1 2026';
    const cleanTitle = stripQuarterPrefix(update.new_title);
    const newTitle = `${quarter}: ${cleanTitle}`;
    const dueDateISO = quarterDueDates[quarter] || null;

    console.log(`Updating milestone #${update.milestone_number}: ${milestone.title} -> ${newTitle}`);

    await github.rest.issues.updateMilestone({
      owner,
      repo,
      milestone_number: update.milestone_number,
      title: newTitle,
      description: update.new_description || milestone.description,
      due_on: dueDateISO
    });
  }

  // 3. Mark stale milestones (close them)
  for (const stale of (actions.stale || [])) {
    const milestone = existingMilestones.find(m => m.number === stale.milestone_number);
    if (!milestone) continue;

    console.log(`Closing stale milestone #${stale.milestone_number}: ${milestone.title} (${stale.reason})`);

    // Query for issues BEFORE closing (GitHub API may not return issues for closed milestones)
    const { data: issues } = await github.rest.issues.listForRepo({
      owner,
      repo,
      milestone: milestone.number,
      state: 'open',
      per_page: 100
    });

    console.log(`Found ${issues.length} open issues assigned to this milestone`);

    // Now close the milestone
    await github.rest.issues.updateMilestone({
      owner,
      repo,
      milestone_number: stale.milestone_number,
      title: milestone.title.startsWith('[stale]') ? milestone.title : `[stale] ${milestone.title}`,
      state: 'closed'
    });

    // Remove milestone from each issue
    for (const issue of issues) {
      console.log(`Removing milestone from issue #${issue.number}`);
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: issue.number,
        body: `The milestone "${milestone.title}" has been marked stale because the corresponding roadmap objective was removed or significantly changed.\n\nPlease review this issue and assign it to a current milestone if still relevant.`
      });

      await github.rest.issues.update({
        owner,
        repo,
        issue_number: issue.number,
        milestone: null
      });
    }
  }

  // 4. Keep unchanged (just log)
  for (const keep of (actions.keep || [])) {
    console.log(`Keeping milestone #${keep.milestone_number} unchanged: ${keep.title}`);
  }

  console.log('Milestone sync complete');
};
