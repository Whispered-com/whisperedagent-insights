'use strict';

/**
 * Insights agent – Express HTTP API.
 *
 * Endpoints:
 *   GET  /           Serve the web UI
 *   POST /start      Start a new conversation for a user
 *   POST /message    Send a message in an ongoing conversation
 *   POST /reset      Reset a user's conversation
 */

require('dotenv').config();

const path = require('path');
const express = require('express');
const AirtableClient = require('./database/airtableClient.js');
const InsightsAgent = require('./agents/insightsAgent.js');
const { StateManager } = require('./agents/state.js');

const PORT = parseInt(process.env.PORT || '3000', 10);

async function main() {
  const db = await AirtableClient.create();
  const stateManager = new StateManager();
  const agent = new InsightsAgent(db, stateManager);

  const app = express();
  app.use(express.json());

  // ── Helpers ──────────────────────────────────────────────────────────────

  function parseBody(body = {}) {
    const userId = (body.user_id || '').trim();
    const userName = (body.user_name || userId).trim();
    let mode = (body.mode || 'premium').trim().toLowerCase();
    if (!['free', 'pro', 'premium'].includes(mode)) mode = 'premium';
    return { userId, userName, mode };
  }

  // ── Routes ───────────────────────────────────────────────────────────────

  app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
  });

  app.post('/start', async (req, res) => {
    const { userId, userName, mode } = parseBody(req.body);
    if (!userId) return res.status(400).json({ error: 'user_id is required' });
    try {
      const reply = await agent.startConversation(userId, userName, mode);
      res.json({ reply });
    } catch (err) {
      console.error('Error in /start:', err);
      res.status(500).json({ reply: `Sorry, something went wrong starting up: ${err.message}` });
    }
  });

  app.post('/message', async (req, res) => {
    const { userId, userName, mode } = parseBody(req.body);
    const text = (req.body.text || '').trim();
    if (!userId || !text) return res.status(400).json({ error: 'user_id and text are required' });

    try {
      // State lost (e.g. server restart) — silently re-create
      if (!stateManager.get(userId)) {
        stateManager.reset(userId, userName, mode);
      }
      const reply = await agent.handleMessage(userId, userName, text, mode);
      const state = stateManager.get(userId);
      const suggestedUpdates = state ? state.suggestedUpdates : {};
      res.json({ reply, suggested_updates: suggestedUpdates });
    } catch (err) {
      console.error('Error in /message:', err);
      res.status(500).json({ reply: `Sorry, something went wrong: ${err.message}` });
    }
  });

  app.post('/reset', async (req, res) => {
    const { userId, userName, mode } = parseBody(req.body);
    if (!userId) return res.status(400).json({ error: 'user_id is required' });
    try {
      const reply = await agent.startConversation(userId, userName, mode);
      res.json({ reply });
    } catch (err) {
      console.error('Error in /reset:', err);
      res.status(500).json({ reply: `Sorry, something went wrong: ${err.message}` });
    }
  });

  // ── Start ─────────────────────────────────────────────────────────────────

  app.listen(PORT, '0.0.0.0', () => {
    console.info(`Insights agent listening on port ${PORT}`);
  });
}

main().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
