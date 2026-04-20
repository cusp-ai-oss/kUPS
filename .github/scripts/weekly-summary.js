/**
 * @fileoverview Weekly summary generator using Claude Sonnet
 * @description Generates stakeholder-friendly progress summaries by analyzing
 * GitHub activity (merged PRs, issues, commits) and connecting to ROADMAP.md.
 * @see .github/workflows/weekly-summary.yml
 */

const fs = require('fs');

/** @type {{model: string, max_tokens: number}} */
const MODEL_CONFIG = JSON.parse(fs.readFileSync('.github/config/models.json', 'utf8')).weekly_summary;

/**
 * Gathers GitHub activity data from the past week
 * @param {Object} github - Octokit REST client
 * @param {Object} context - GitHub Actions context with repo info
 * @returns {Promise<Object>} Activity data including merged PRs, issues, and contributors
 */
async function gatherActivity(github, context) {
  const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
  const { owner, repo } = context.repo;

  const [prs, closedIssues, newIssues, commits, milestones] = await Promise.all([
    github.rest.pulls.list({ owner, repo, state: 'closed', sort: 'updated', direction: 'desc', per_page: 50 }),
    github.rest.issues.listForRepo({ owner, repo, state: 'closed', since: oneWeekAgo, per_page: 50 }),
    github.rest.issues.listForRepo({ owner, repo, state: 'all', since: oneWeekAgo, per_page: 50 }),
    github.rest.repos.listCommits({ owner, repo, since: oneWeekAgo, per_page: 100 }),
    github.rest.issues.listMilestones({ owner, repo, state: 'open', per_page: 20 })
  ]);

  // Filter to PRs merged this week
  const mergedThisWeek = prs.data.filter(pr =>
    pr.merged_at && new Date(pr.merged_at) > new Date(oneWeekAgo)
  );

  // Fetch rich details for each merged PR (in parallel)
  const mergedPRs = await Promise.all(mergedThisWeek.map(async pr => {
    const [files, mergeCommit] = await Promise.all([
      github.rest.pulls.listFiles({ owner, repo, pull_number: pr.number }),
      pr.merge_commit_sha
        ? github.rest.git.getCommit({ owner, repo, commit_sha: pr.merge_commit_sha })
        : null
    ]);
    return {
      title: pr.title,
      number: pr.number,
      author: pr.user.login,
      labels: pr.labels.map(l => l.name),
      body: pr.body ? pr.body.slice(0, 500) : '', // Truncate for prompt size
      mergeCommitMessage: mergeCommit?.data?.message?.slice(0, 1000) || '', // Squash commit message
      additions: pr.additions,
      deletions: pr.deletions,
      changedFiles: files.data.slice(0, 10).map(f => f.filename) // Top 10 files
    };
  }));

  return {
    mergedPRs,
    closedIssues: closedIssues.data
      .filter(i => !i.pull_request)
      .map(i => ({ title: i.title, number: i.number, labels: i.labels.map(l => l.name) })),
    openedIssues: newIssues.data
      .filter(i => !i.pull_request && new Date(i.created_at) > new Date(oneWeekAgo))
      .map(i => ({ title: i.title, number: i.number, labels: i.labels.map(l => l.name) })),
    milestones: milestones.data.map(m => ({
      title: m.title,
      open_issues: m.open_issues,
      closed_issues: m.closed_issues
    })),
    commitCount: commits.data.length,
    contributors: [...new Set(commits.data.map(c => c.author?.login).filter(Boolean))]
  };
}

/**
 * Calls Claude API with a prompt
 */
async function callClaude(prompt, apiKey) {
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

  return result.content[0].text;
}

/**
 * Generates a short Slack teaser from activity
 */
async function generateSlackTeaser(activity, apiKey) {
  const activitySummary = `PRs merged: ${activity.mergedPRs.length} (${activity.mergedPRs.map(p => p.title).join(', ')})
Issues closed: ${activity.closedIssues.length}
Issues opened: ${activity.openedIssues.length}
Contributors: ${activity.contributors.join(', ')}`;

  let prompt = fs.readFileSync('.github/prompts/slack-teaser.md', 'utf8');
  prompt = prompt.replace('{{ACTIVITY_SUMMARY}}', activitySummary);

  return callClaude(prompt, apiKey);
}

/**
 * Calls Claude API to generate a summary from activity data
 */
async function generateSummary(activity, apiKey) {
  let prompt = fs.readFileSync('.github/prompts/weekly-summary.md', 'utf8');
  prompt = prompt.replace('{{ACTIVITY_DATA}}', JSON.stringify(activity, null, 2));

  // Include roadmap context if available
  try {
    const roadmap = fs.readFileSync('ROADMAP.md', 'utf8');
    prompt = prompt.replace('{{ROADMAP}}', roadmap);
  } catch {
    prompt = prompt.replace('{{ROADMAP}}', '(No roadmap available)');
  }

  return callClaude(prompt, apiKey);
}

/**
 * Main entry point for GitHub Actions
 * @param {Object} params - GitHub Actions context
 * @param {Object} params.github - Octokit REST client
 * @param {Object} params.context - GitHub Actions context with repo info
 * @param {Object} params.core - GitHub Actions core utilities for outputs
 */
module.exports = async ({ github, context, core }) => {
  const activity = await gatherActivity(github, context);

  // Skip if no activity
  if (activity.mergedPRs.length === 0 && activity.closedIssues.length === 0 && activity.openedIssues.length === 0) {
    console.log('No activity this week, skipping summary');
    core.setOutput('skip', 'true');
    return;
  }

  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    const [summary, slackTeaser] = await Promise.all([
      generateSummary(activity, apiKey),
      generateSlackTeaser(activity, apiKey)
    ]);
    core.setOutput('skip', 'false');
    core.setOutput('summary', summary);
    core.setOutput('slack_teaser', slackTeaser);
  } catch (e) {
    console.error('Summary generation failed:', e.message);
    core.setOutput('skip', 'true');
  }
};
