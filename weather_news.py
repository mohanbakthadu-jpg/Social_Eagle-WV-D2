import os
import asyncio
from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()


def search_weather_news(city: str) -> str:
    """
    Search the web for today's weather news and forecast updates
    for a given city. Use this tool whenever a user asks about
    weather, rain, storms, temperature, forecast, or weather news.
    """

    results = []

    with DDGS() as ddgs:
        search_results = ddgs.text(
            f"{city} weather today news",
            max_results=5
        )

        for item in search_results:
            title = item.get("title", "")
            body = item.get("body", "")

            results.append(f"{title} - {body}")

    return "\n".join(results)


async def main():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables."
        )

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )

    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        system_message=(
            "You are a weather news assistant. "
            "Always answer in exactly 3 short lines."
        ),
        tools=[search_weather_news],
        reflect_on_tool_use=True
    )

    city = input("Enter a city: ")

    response = await weather_agent.run(
        task=f"Give today's weather news for {city} in exactly 3 short lines."
    )

    print("\nWeather News:\n")
    print(response.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())