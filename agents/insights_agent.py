"""
Insights Agent – core conversation logic (read-only mode).

Flow:
1. User names a company / role
2. Agent looks it up in Airtable and generates a synopsis via Claude
3. Agent answers follow-up questions about the company/role
"""

import json
import logging
import os
from typing import Optional

import anthropic

from database.airtable_client import AirtableClient
from agents.state import ConversationState, Phase, StateManager
from prompts.synopsis import build_company_synopsis_prompt, build_role_synopsis_prompt

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the Insights agent for a professional community platform that tracks companies and open roles.
Your goal is to help community members quickly learn what we know about a company or role they are interested in.

Be warm, concise, and professional. Respond conversationally – you are a knowledgeable, helpful colleague, not a form.
When you need to identify a company or role, extract the name clearly from natural language.
"""


class InsightsAgent:
    def __init__(self, db: AirtableClient, state_manager: StateManager):
        self.db = db
        self.state_manager = state_manager
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle_message(self, user_id: str, user_name: str, user_text: str) -> str:
        """Process one turn of conversation and return the agent's reply."""
        state = self.state_manager.get_or_create(user_id, user_name)
        state.add_user_message(user_text)

        if state.phase == Phase.IDENTIFY:
            reply = self._handle_identify(state, user_text)
        elif state.phase in (Phase.COMPANY_FOUND, Phase.ROLE_FOUND):
            reply = self._handle_followup(state, user_text)
        else:
            # Reset so they can start a new lookup
            state.phase = Phase.IDENTIFY
            reply = self._handle_identify(state, user_text)

        state.add_assistant_message(reply)
        return reply

    def start_conversation(self, user_id: str, user_name: str) -> str:
        """Called when the user first opens the agent."""
        state = self.state_manager.reset(user_id, user_name)
        greeting = (
            f"Hey {user_name}! I'm the Insights agent.\n\n"
            "Tell me which company or role you want to know about and I'll pull up everything "
            "we have on it.\n\n"
            "For example: \"Airtable\" or \"RevOps Manager at Salesforce\"."
        )
        state.add_assistant_message(greeting)
        return greeting

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_identify(self, state: ConversationState, user_text: str) -> str:
        """Parse company/role from user message and look them up."""
        parsed = self._parse_company_and_role(user_text)
        company_name = parsed.get("company")
        role_title = parsed.get("role")

        if not company_name and not role_title:
            return (
                "I didn't catch a company or role name there. "
                "Could you try again? For example: \"Acme Corp\" or \"Product Manager at Acme Corp\"."
            )

        company_record = self.db.find_company(company_name) if company_name else None

        role_record = None
        if role_title:
            role_record = self.db.find_role(role_title, company_record["id"] if company_record else None)
            if role_record and not company_record:
                # Resolve company from the role's linked record
                linked = role_record["fields"].get("Company", [])
                if linked:
                    company_record = self.db.get_company(linked[0])

        # Both found
        if company_record and role_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Name", company_name)
            state.role_record_id = role_record["id"]
            state.role_title = role_record["fields"].get("Title", role_title)
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(company_record, role_record)

        # Company only
        if company_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Name", company_name)
            state.phase = Phase.COMPANY_FOUND
            return self._generate_company_synopsis(company_record, role_title)

        # Nothing found
        entity = company_name or role_title
        return (
            f"I don't have \"{entity}\" in our database yet. "
            "You could try a slightly different name, or ask me about a different company or role."
        )

    def _handle_followup(self, state: ConversationState, user_text: str) -> str:
        """Answer a follow-up question using the full conversation history."""
        # If the user seems to be asking about a different company, reset
        parsed = self._parse_company_and_role(user_text)
        new_company = parsed.get("company")
        if new_company and new_company.lower() != (state.company_name or "").lower():
            state.phase = Phase.IDENTIFY
            return self._handle_identify(state, user_text)

        return self._call_claude(state.messages)

    # ------------------------------------------------------------------
    # Synopsis generators
    # ------------------------------------------------------------------

    def _generate_company_synopsis(self, company_record: dict, role_hint: Optional[str]) -> str:
        roles = self.db.get_company_roles(company_record["id"])
        prompt = build_company_synopsis_prompt(company_record, roles, [])
        synopsis = self._call_claude([{"role": "user", "content": prompt}])

        suffix = "\n\nWhat would you like to know more about? You can ask me anything or look up a specific role."
        if role_hint:
            suffix += f" (Did you mean the \"{role_hint}\" role specifically?)"
        return synopsis + suffix

    def _generate_role_synopsis(self, company_record: Optional[dict], role_record: dict) -> str:
        prompt = build_role_synopsis_prompt(role_record, company_record or {}, [])
        synopsis = self._call_claude([{"role": "user", "content": prompt}])
        return synopsis + "\n\nAnything else you'd like to know about this role or company?"

    # ------------------------------------------------------------------
    # Claude helpers
    # ------------------------------------------------------------------

    def _parse_company_and_role(self, user_text: str) -> dict:
        """Ask Claude to extract company name and role title from free-form text."""
        prompt = (
            "Extract the company name and job role title from this message. "
            "Return a JSON object with keys 'company' and 'role'. "
            "Use null for any field you cannot determine.\n\n"
            f"Message: {user_text}\n\nJSON:"
        )
        raw = self._call_claude([{"role": "user", "content": prompt}])
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"company": None, "role": None}

    def _call_claude(self, messages: list[dict], max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
