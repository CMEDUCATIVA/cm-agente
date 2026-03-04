from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.revit_plugin_tools import revit_plugin_tools
from core import get_model, settings


class AgentState(MessagesState, total=False):
    pass


current_date = datetime.now().strftime("%B %d, %Y")
MAX_HISTORY_MESSAGES = 14
instructions = f"""
You are Sara-Diseño-GPT, an autonomous assistant specialized in Autodesk Revit workflows.
Today's date is {current_date}.

Rules:
- Answer in the user's language (Spanish if user writes in Spanish).
- Prefer tool use over assumptions when model state is required.
- Before destructive actions (Delete), explicitly confirm intent with the user.
- Keep responses concise and execution-focused.
- After each tool call, summarize what changed and include element IDs when available.
- If plugin connectivity fails, explain the exact failure and ask user to open Revit plugin service.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(revit_plugin_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"][-MAX_HISTORY_MESSAGES:],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await wrap_model(model).ainvoke(state, config)
    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(revit_plugin_tools))
agent.set_entry_point("model")


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")
sara_diseno_gpt = agent.compile()

