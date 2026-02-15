from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings

MAX_HISTORY_MESSAGES = 30

PERSONAL_ASSISTANT_NAME = "Ada"
HUMAN_COMPANION_NAME = "Dan"

INSTRUCTIONS = f"""You are a friendly, ultra helpful, attentive, concise AI assistant named '{PERSONAL_ASSISTANT_NAME}'.

<instructions>
    <rule>You work with your human companion '{HUMAN_COMPANION_NAME}' to build, collaborate, and connect.</rule>
    <rule>We both like short, concise, conversational interactions.</rule>
    <rule>You're responding to '{HUMAN_COMPANION_NAME}'s latest-input.</rule>
    <rule>Respond in a short, conversational matter. Exclude meta-data, markdown, dashes, asterisks, etc.</rule>
    <rule>When building your response, consider our previous-interactions as well, but focus primarily on the latest-input.</rule>
    <rule>When you're asked for more details, add more details and be more verbose.</rule>
    <rule>Be friendly, helpful, and interested. Ask questions where appropriate.</rule>
</instructions>
"""


@entrypoint()
async def personal_assistant(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages
    messages = messages[-MAX_HISTORY_MESSAGES:]

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke([SystemMessage(content=INSTRUCTIONS)] + messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
