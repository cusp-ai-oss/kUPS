/**
 * @fileoverview Shared Claude API call for the GitHub Actions scripts.
 * @description Reads each task's model, effort and token limit from
 * .github/config/models.json and returns the text of the reply.
 */

const fs = require('fs');

const MODELS = JSON.parse(fs.readFileSync('.github/config/models.json', 'utf8'));

/**
 * Sends one prompt for a configured task and returns the reply text.
 * @param {string} task - Key in models.json, e.g. "issue_labeling"
 * @param {string} prompt - The user message
 * @param {string} apiKey - Anthropic API key
 * @returns {Promise<string>} Concatenated text blocks of the reply
 * @throws {Error} On an API error, a refusal, or a reply without text
 */
async function callClaude(task, prompt, apiKey) {
  const { model, max_tokens, effort, fallbacks } = MODELS[task];
  const headers = {
    'x-api-key': apiKey,
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json'
  };
  const body = {
    model,
    max_tokens,
    output_config: { effort },
    messages: [{ role: 'user', content: prompt }]
  };
  if (fallbacks) {
    headers['anthropic-beta'] = 'server-side-fallback-2026-07-01';
    body.fallbacks = fallbacks;
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers,
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (result.error) {
    throw new Error(`API error: ${result.error.message}`);
  }
  if (result.stop_reason === 'refusal') {
    throw new Error('Claude declined the request');
  }

  // Thinking blocks come first on current models; keep only the answer.
  const text = result.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
    .trim();
  if (!text) {
    throw new Error(`Reply has no text (stop_reason: ${result.stop_reason})`);
  }
  return text;
}

module.exports = { callClaude };
