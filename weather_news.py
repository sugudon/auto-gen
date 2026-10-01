import asyncio
import os

from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Read the API key from the environment, never hard-code it.
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")


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


city = input("Which city? ").strip()
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
    system_message=(
        "You are a concise weather news reporter. Always call search_weather_news "
        "for the city you are given, then summarise only what the results say. "
        "Reply in exactly 3 short lines, one sentence per line, with no bullets, "
        "numbering, headings or extra text."
    ),
    reflect_on_tool_use=True,  # write an answer from the results, not a raw dump
)


async def main():
    try:
        await Console(
            agent.run_stream(
                task=f"Give me today's weather news for {city} in exactly 3 short lines."
            )
        )
    finally:
        await model_client.close()


asyncio.run(main())
