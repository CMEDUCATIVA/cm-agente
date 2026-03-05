from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
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
- For any request that references a `Revit_*` tool or asks for Revit model data, call the tool first.
- Never fabricate tool outputs, IDs, counts, categories, or JSON data.
- If a required tool call cannot be executed, respond exactly: TOOL_CALL_FAILED plus one short reason.
- Before destructive actions (Delete), explicitly confirm intent with the user.
- Keep responses concise and execution-focused.
- After each tool call, summarize what changed and include element IDs when available.
- If plugin connectivity fails, explain the exact failure and ask user to open Revit plugin service.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(revit_plugin_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)]
        + _drop_incomplete_tool_call_turns(state["messages"][-MAX_HISTORY_MESSAGES:]),
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def _drop_incomplete_tool_call_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove malformed assistant tool-call turns that have missing ToolMessage responses."""
    cleaned: list[BaseMessage] = []
    pending_ids: set[str] = set()
    pending_block: list[BaseMessage] = []

    for msg in messages:
        if not pending_ids:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                pending_ids = {str(tc.get("id") or "") for tc in msg.tool_calls if tc.get("id")}
                pending_block = [msg]
                # If IDs are missing, the turn is already malformed; drop it.
                if not pending_ids:
                    pending_block = []
            else:
                cleaned.append(msg)
            continue

        if isinstance(msg, ToolMessage) and msg.tool_call_id in pending_ids:
            pending_block.append(msg)
            pending_ids.remove(msg.tool_call_id)
            if not pending_ids:
                cleaned.extend(pending_block)
                pending_block = []
            continue

        # Unexpected message before all tool IDs are resolved: drop the malformed turn.
        pending_ids = set()
        pending_block = []
        if isinstance(msg, AIMessage) and msg.tool_calls:
            pending_ids = {str(tc.get("id") or "") for tc in msg.tool_calls if tc.get("id")}
            pending_block = [msg] if pending_ids else []
        else:
            cleaned.append(msg)

    # If we end with unresolved IDs, drop that incomplete turn.
    return cleaned


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await wrap_model(model).ainvoke(state, config)
    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(revit_plugin_tools, handle_tool_errors=True))
agent.set_entry_point("model")


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")
sara_diseno_gpt = agent.compile()
