import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Read the API key from the environment, never hard-code it.
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

agent = AssistantAgent(
    name="hr_assistant",
    model_client=model_client,
    system_message="You are an HR assistant.",
)

# "5 lines" is ambiguous, so the task says exactly what is wanted:
# a title line followed by exactly five numbered lines, one sentence each.
TASK = (
    "Write a job description for a Junior Python Developer. "
    "Output the job title on the first line, then exactly 5 numbered lines "
    "(1. to 5.), each a single sentence covering responsibilities or requirements. "
    "No intro, no closing text, nothing else."
)


async def main():
    try:
        await Console(agent.run_stream(task=TASK))
    finally:
        # An unclosed client leaves the script hanging.
        await model_client.close()


asyncio.run(main())
