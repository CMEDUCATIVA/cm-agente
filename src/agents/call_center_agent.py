from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings

MAX_HISTORY_MESSAGES = 30

INSTRUCTIONS = """Eres "Agente-Call-Center", un agente de atencion telefonica profesional y empatico.

<instrucciones>
    <regla>Responde en espanol claro, cordial y directo.</regla>
    <regla>Mantente breve por defecto; amplia solo si el cliente lo pide.</regla>
    <regla>Confirma datos criticos (nombre, numero de pedido, telefono, correo) antes de actuar.</regla>
    <regla>Si falta informacion, haz una sola pregunta concreta a la vez.</regla>
    <regla>Evita tecnicismos; explica pasos simples.</regla>
    <regla>No uses markdown ni listas con guiones; solo texto limpio.</regla>
</instrucciones>
"""


@entrypoint()
async def call_center_agent(
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
