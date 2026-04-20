/**
 * @fileoverview PR labeling using Claude
 * @description Auto-applies area labels to PRs based on changed files.
 * @see .github/workflows/pr-labeling.yml
 */

const fs = require('fs');

/** @type {{model: string, max_tokens: number}} */
const MODEL_CONFIG = JSON.parse(fs.readFileSync('.github/config/models.json', 'utf8')).pr_review;

/**
 * Calls Claude API to label a PR
 */
async function labelPR(prompt, apiKey) {
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
 */
module.exports = async ({ github, context }) => {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const pr = context.payload.pull_request;

  if (!pr) {
    console.log('No PR found in context');
    return;
  }

  // Get changed files
  const { data: files } = await github.rest.pulls.listFiles({
    owner, repo,
    pull_number: pr.number,
    per_page: 100
  });
  const changedFiles = files.map(f => f.filename).join('\n');

  // Get existing PR labels
  const existingLabels = pr.labels.map(l => l.name);

  // Get available labels from repo
  const { data: repoLabels } = await github.rest.issues.listLabelsForRepo({
    owner, repo, per_page: 100
  });
  const availableLabels = repoLabels.map(l => l.name);

  // Build prompt
  let prompt = fs.readFileSync('.github/prompts/pr-labeling.md', 'utf8');
  prompt = prompt.replace('{{PR_TITLE}}', pr.title);
  prompt = prompt.replace('{{CHANGED_FILES}}', changedFiles || '(none)');
  prompt = prompt.replace('{{AVAILABLE_LABELS}}', JSON.stringify(availableLabels, null, 2));
  prompt = prompt.replace('{{EXISTING_LABELS}}', JSON.stringify(existingLabels, null, 2));

  // Get labels from Claude
  let result;
  try {
    result = await labelPR(prompt, process.env.ANTHROPIC_API_KEY);
  } catch (e) {
    console.error('Labeling failed:', e.message);
    return;
  }

  // Apply labels (only add missing ones)
  const labelsToAdd = (result.labels || []).filter(label =>
    availableLabels.includes(label) && !existingLabels.includes(label)
  );

  if (labelsToAdd.length > 0) {
    console.log(`Adding labels: ${labelsToAdd.join(', ')}`);
    await github.rest.issues.addLabels({
      owner, repo,
      issue_number: pr.number,
      labels: labelsToAdd
    });
  } else {
    console.log('No new labels to add');
  }
};
