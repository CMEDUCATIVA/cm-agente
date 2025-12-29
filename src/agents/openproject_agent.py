from datetime import datetime
import ast
import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.openproject_tools import openproject_tools
from core import get_model, settings


class AgentState(MessagesState, total=False):
    pass


current_date = datetime.now().strftime("%B %d, %Y")
MAX_HISTORY_MESSAGES = 10
instructions = f"""
You are an OpenProject assistant. You can read and update OpenProject data by calling tools.
Today's date is {current_date}.

Rules:
- Answer in the user's language (Spanish if they speak Spanish).
- Prefer asking a clarifying question if the user hasn't specified a project.
- When creating or updating items, confirm the exact fields you will set.
- Summarize results clearly and include IDs for created/updated entities.
- If you list projects and the result is truncated, ask the user for a project name and use the search tool.
- If a tool response contains a `rendered` field, you MUST reply with that `rendered` text verbatim (do not reformat, do not omit fields).
- If the user asks to list paquetes de trabajo, tareas, work packages, or information about work_packages, you MUST call OpenProject_ListWorkPackages.
- When calling OpenProject_ListWorkPackages, always include user_question with the user's request verbatim.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(openproject_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"][-MAX_HISTORY_MESSAGES:],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await wrap_model(m).ainvoke(state, config)
    return {"messages": [response]}


def _extract_rendered_from_tool_message(message: ToolMessage) -> str | None:
    content = message.content
    data: dict | None = None
    if isinstance(content, dict):
        data = content
    elif isinstance(content, str) and content.strip():
        try:
            data = json.loads(content)
        except Exception:
            try:
                data = ast.literal_eval(content)
            except Exception:
                return None
    else:
        return None

    if isinstance(data, dict):
        rendered = data.get("rendered")
        if isinstance(rendered, str) and rendered.strip():
            return rendered.strip()
        analysis = data.get("analysis")
        download_url = data.get("download_url")
        if isinstance(analysis, str) and isinstance(download_url, str):
            return f"{analysis}\n\nDescarga el reporte completo aqui: {download_url}"
    return None


async def arender_from_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            rendered = _extract_rendered_from_tool_message(msg)
            if rendered:
                return {"messages": [AIMessage(content=rendered)]}
    return {"messages": []}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(openproject_tools))
agent.add_node("render", arender_from_tools)
agent.set_entry_point("model")


def after_tools(state: AgentState) -> Literal["render", "model"]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            rendered = _extract_rendered_from_tool_message(msg)
            if rendered:
                return "render"
            break
    return "model"


agent.add_conditional_edges("tools", after_tools, {"render": "render", "model": "model"})


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("render", END)
openproject_agent = agent.compile()
