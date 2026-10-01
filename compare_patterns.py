"""Compare AutoGen orchestration patterns on the same task: 3-line weather news.

Usage:
    export OPENAI_API_KEY="your-key"
    python compare_patterns.py Chennai                  # run every pattern
    python compare_patterns.py Chennai single graphflow # run only some

Patterns: single, custom, selector, swarm, graphflow, magentic
Each run reports wall time, LLM tokens, number of messages and whether the
final answer really is 3 lines, so "which is best" is measured, not guessed.
"""

import argparse
import asyncio
import os
import time
from typing import Sequence

from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import SourceMatchTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.teams import (
    DiGraphBuilder,
    GraphFlow,
    MagenticOneGroupChat,
    RoundRobinGroupChat,
    SelectorGroupChat,
    Swarm,
)
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

THREE_LINES = (
    "Reply in exactly 3 short lines, one sentence per line, with no bullets, "
    "numbering, headings or extra text."
)


def search_weather_news(city: str) -> str:
    """Search the web for TODAY's weather news for a city.

    Use this tool whenever the user asks about current weather, rain, heat,
    storms, forecasts or weather alerts in a specific city. It returns recent
    news headlines and snippets about the weather there right now. Do not use
    it for general trivia or non-weather questions.

    Args:
        city: Name of the city, e.g. "Chennai".

    Returns:
        Raw news search results (titles, dates, sources, snippets) as text.
    """
    return str(DDGS().news(query=f"{city} weather today", max_results=5))


# ---------------------------------------------------------------- agents
def make_single(mc, city):
    """One agent + one tool. The baseline."""
    agent = AssistantAgent(
        name="weather_news",
        model_client=mc,
        tools=[search_weather_news],
        system_message=(
            "You are a concise weather news reporter. Always call "
            f"search_weather_news first, then summarise only what it returns. {THREE_LINES}"
        ),
        reflect_on_tool_use=True,
    )
    return agent  # a single agent is run directly, not as a team


def make_writer(mc):
    return AssistantAgent(
        name="writer",
        model_client=mc,
        description="Writes the final 3-line weather news from the search results.",
        system_message=(
            "You are a concise weather news reporter. Using ONLY the search "
            f"results already in the conversation, write the final answer. {THREE_LINES}"
        ),
    )


def make_researcher(mc, handoffs=None):
    return AssistantAgent(
        name="researcher",
        model_client=mc,
        description="Searches the web for today's weather news for the city.",
        tools=[search_weather_news],
        handoffs=handoffs,
        system_message=(
            "You are a researcher. Call search_weather_news for the city in the "
            "task. Do not write the final answer."
            + (" After the search, hand off to writer." if handoffs else "")
        ),
    )


class WeatherSearchAgent(BaseChatAgent):
    """Custom agent: no LLM at all, it just runs the search tool in code.

    Deterministic and free for the search step; only the writer costs tokens.
    """

    def __init__(self, city: str):
        super().__init__("searcher", "Runs the web search for today's weather news.")
        self._city = city

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token: CancellationToken) -> Response:
        # DDGS is blocking, so keep it off the event loop.
        found = await asyncio.to_thread(search_weather_news, self._city)
        return Response(chat_message=TextMessage(content=found, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


# ---------------------------------------------------------------- teams
def make_custom(mc, city):
    """Custom agent + writer in a fixed round-robin order."""
    return RoundRobinGroupChat(
        [WeatherSearchAgent(city), make_writer(mc)],
        termination_condition=SourceMatchTermination(["writer"]),
    )


def make_selector(mc, city):
    """An LLM picks who speaks next."""
    return SelectorGroupChat(
        [make_researcher(mc), make_writer(mc)],
        model_client=mc,
        termination_condition=SourceMatchTermination(["writer"]),
        selector_prompt=(
            "Roles:\n{roles}\n\nConversation so far:\n{history}\n\n"
            "Pick the next speaker from {participants}. The researcher must go "
            "first to search; once search results exist, the writer goes. "
            "Return only the role name."
        ),
    )


def make_swarm(mc, city):
    """Agents hand control to each other explicitly via handoffs."""
    return Swarm(
        [make_researcher(mc, handoffs=["writer"]), make_writer(mc)],
        termination_condition=SourceMatchTermination(["writer"]),
    )


def make_graphflow(mc, city):
    """A fixed directed graph: researcher -> writer. No LLM routing."""
    researcher, writer = make_researcher(mc), make_writer(mc)
    g = DiGraphBuilder()
    g.add_node(researcher).add_node(writer)
    g.add_edge(researcher, writer)
    g.set_entry_point(researcher)
    return GraphFlow(participants=g.get_participants(), graph=g.build())


def make_magentic(mc, city):
    """Magentic-One orchestrator plans, tracks a ledger, and delegates."""
    return MagenticOneGroupChat(
        [make_researcher(mc), make_writer(mc)], model_client=mc, max_turns=12
    )


PATTERNS = {
    "single": make_single,
    "custom": make_custom,
    "selector": make_selector,
    "swarm": make_swarm,
    "graphflow": make_graphflow,
    "magentic": make_magentic,
}


# ---------------------------------------------------------------- runner
def final_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, TextMessage) and m.source != "user":
            return m.content.strip()
    return ""


async def run_pattern(name: str, mc, city: str) -> dict:
    task = f"Give me today's weather news for {city} in exactly 3 short lines."
    runner = PATTERNS[name](mc, city)
    start = time.perf_counter()
    try:
        result = await runner.run(task=task)
        error = None
    except Exception as e:  # keep going so one failure doesn't hide the rest
        result, error = None, f"{type(e).__name__}: {e}"
    secs = time.perf_counter() - start
    if result is None:
        return {"pattern": name, "error": error, "secs": secs}
    tokens = sum(
        (m.models_usage.prompt_tokens + m.models_usage.completion_tokens)
        for m in result.messages
        if getattr(m, "models_usage", None)
    )
    answer = final_text(result.messages)
    lines = [ln for ln in answer.splitlines() if ln.strip()]
    return {
        "pattern": name,
        "secs": secs,
        "tokens": tokens,
        "messages": len(result.messages),
        "lines": len(lines),
        "answer": answer,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("city")
    parser.add_argument("patterns", nargs="*", metavar="pattern",
                        help=f"any of: {', '.join(PATTERNS)} (default: all)")
    args = parser.parse_args()
    bad = [p for p in args.patterns if p not in PATTERNS]
    if bad:
        parser.error(f"unknown pattern(s): {', '.join(bad)}")
    chosen = args.patterns or list(PATTERNS)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")
    mc = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    rows = []
    try:
        for name in chosen:
            print(f"\n=== {name} ===")
            row = await run_pattern(name, mc, args.city)
            rows.append(row)
            print(row.get("answer") or f"FAILED: {row.get('error')}")
    finally:
        await mc.close()

    print("\n" + "pattern".ljust(10) + "secs".rjust(7) + "tokens".rjust(9)
          + "msgs".rjust(6) + "lines".rjust(7) + "  ok(3 lines)")
    for r in rows:
        if "error" in r:
            print(r["pattern"].ljust(10) + f"{r['secs']:7.1f}  FAILED: {r['error']}")
            continue
        ok = "yes" if r["lines"] == 3 else "NO"
        print(r["pattern"].ljust(10) + f"{r['secs']:7.1f}{r['tokens']:9d}"
              + f"{r['messages']:6d}{r['lines']:7d}  {ok}")


if __name__ == "__main__":
    asyncio.run(main())
