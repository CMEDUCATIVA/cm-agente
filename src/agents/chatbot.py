from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings

MAX_HISTORY_MESSAGES = 10


@entrypoint()
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages
    messages = messages[-MAX_HISTORY_MESSAGES:]

    extra = config["configurable"].get("instructions")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    if extra:
        response = await model.ainvoke(
            [SystemMessage(content=f"<instrucciones_usuario>\n{extra}\n</instrucciones_usuario>")]
            + messages
        )
    else:
        response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
