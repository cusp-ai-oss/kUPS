/**
 * @fileoverview Issue labeling using Claude Haiku
 * @description Automatically classifies and labels new GitHub issues based on
 * content analysis. Labels are extracted from MANIFESTO.md (source of truth).
 * Optionally assigns issues to roadmap-aligned milestones.
 * @see .github/workflows/issue-labeling.yml
 */

const fs = require('fs');

/** @type {{model: string, max_tokens: number}} */
const MODEL_CONFIG = JSON.parse(fs.readFileSync('.github/config/models.json', 'utf8')).issue_labeling;

/**
 * Calls Claude API to classify an issue
 * @param {string} prompt - The formatted prompt containing issue details
 * @param {string} apiKey - Anthropic API key
 * @returns {Promise<Object>} Classification result with type, priority, area, and optional milestone
 * @throws {Error} If API returns an error response
 */
async function classifyIssue(prompt, apiKey) {
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
 * @param {Object} params.context - GitHub Actions context with repo and payload info
 */
module.exports = async ({ github, context }) => {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  // Get issue from event payload or fetch by number (for workflow_dispatch)
  let issue = context.payload.issue;
  if (!issue && context.payload.inputs?.issue_number) {
    const { data } = await github.rest.issues.get({
      owner, repo,
      issue_number: parseInt(context.payload.inputs.issue_number, 10)
    });
    issue = data;
  }

  if (!issue) {
    console.log('No issue found in context');
    return;
  }

  // Detect which label categories are already present
  const existingLabels = issue.labels.map(l => l.name);
  const hasType = existingLabels.some(l => l.startsWith('type:'));
  const hasPriority = existingLabels.some(l => l.startsWith('priority:'));
  const hasArea = existingLabels.some(l => l.startsWith('area:'));
  const hasMilestone = !!issue.milestone;

  console.log(`Issue #${issue.number} status: type=${hasType}, priority=${hasPriority}, area=${hasArea}, milestone=${hasMilestone}`);

  // Skip only if fully labeled AND has milestone
  if (hasType && hasPriority && hasArea && hasMilestone) {
    console.log('Issue fully labeled with milestone, skipping');
    return;
  }

  // Fetch available labels from GitHub (raw, no processing)
  const { data: repoLabels } = await github.rest.issues.listLabelsForRepo({
    owner, repo, per_page: 100
  });
  const availableLabels = repoLabels.map(l => l.name);

  // Read MANIFESTO for semantic context (raw, no processing)
  let manifesto = '';
  try {
    manifesto = fs.readFileSync('MANIFESTO.md', 'utf8');
  } catch {
    console.log('Warning: Could not read MANIFESTO.md');
  }

  // Fetch current open milestones
  const { data: milestones } = await github.rest.issues.listMilestones({
    owner, repo, state: 'open', per_page: 100
  });

  const milestoneInfo = milestones.map(m => ({
    title: m.title,
    number: m.number,
    description: m.description || ''
  }));

  // Build prompt
  let prompt = fs.readFileSync('.github/prompts/issue-labeling.md', 'utf8');
  prompt = prompt.replace('{{ISSUE_TITLE}}', issue.title);
  prompt = prompt.replace('{{ISSUE_BODY}}', issue.body || '(no body)');
  prompt = prompt.replace('{{MILESTONES}}', JSON.stringify(milestoneInfo, null, 2));
  prompt = prompt.replace('{{LABELS}}', JSON.stringify(availableLabels, null, 2));
  prompt = prompt.replace('{{MANIFESTO}}', manifesto);

  // Classify
  let classification;
  try {
    classification = await classifyIssue(prompt, process.env.ANTHROPIC_API_KEY);
  } catch (e) {
    console.error('Classification failed:', e.message);
    return;
  }

  // Only add labels for missing categories
  const labelsToAdd = (classification.labels || []).filter(label => {
    if (label.startsWith('type:') && hasType) return false;
    if (label.startsWith('priority:') && hasPriority) return false;
    if (label.startsWith('area:') && hasArea) return false;
    return true;
  });

  if (labelsToAdd.length > 0) {
    console.log(`Adding labels: ${labelsToAdd.join(', ')}`);
    await github.rest.issues.addLabels({
      owner, repo,
      issue_number: issue.number,
      labels: labelsToAdd
    });
  } else {
    console.log('No labels to apply (all categories present)');
  }

  // Assign milestone only if not already set
  if (classification.milestone && !hasMilestone) {
    const milestone = milestoneInfo.find(m => m.title === classification.milestone);
    if (milestone) {
      console.log(`Assigning milestone: ${milestone.title}`);
      await github.rest.issues.update({
        owner, repo,
        issue_number: issue.number,
        milestone: milestone.number
      });
    }
  }

  // Add comment if there's reasoning (only for new issues without any labels)
  if (classification.reasoning && existingLabels.length === 0) {
    await github.rest.issues.createComment({
      owner, repo,
      issue_number: issue.number,
      body: `**Roadmap alignment:** ${classification.reasoning}`
    });
  }

  console.log(`Processed issue #${issue.number}:`, classification);
};
