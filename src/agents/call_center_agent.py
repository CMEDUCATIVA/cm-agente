from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings

MAX_HISTORY_MESSAGES = 30

INSTRUCTIONS = """Eres "Agente-Call-Center", un agente de atencion telefonica profesional y empatico.

<instrucciones>
    <regla>Responde en espanol claro, cordial y alegre, con voz sonriente y cercana.</regla>
    <regla>Usa frases positivas y un ritmo amable.</regla>
    <regla>Siempre inicia la respuesta con "Genial!" de forma entusiasta.</regla>
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

    extra = config["configurable"].get("instructions")
    system_text = INSTRUCTIONS
    if extra:
        system_text = f"{INSTRUCTIONS}\n\n<instrucciones_usuario>\n{extra}\n</instrucciones_usuario>"
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke([SystemMessage(content=system_text)] + messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
