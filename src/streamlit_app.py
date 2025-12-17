import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"
OPENPROJECT_AGENT_ID = "openproject-agent"


def _openproject_mcp_base_url() -> str | None:
    base = os.getenv("OPENPROJECT_MCP_URL")
    if not base:
        return None
    return base.rstrip("/")


def _openproject_mcp_auth_headers() -> dict[str, str]:
    enabled = os.getenv("OPENPROJECT_MCP_HTTP_AUTH_ENABLED", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    if not enabled:
        return {}
    username = os.getenv("OPENPROJECT_MCP_HTTP_AUTH_USERNAME")
    password = os.getenv("OPENPROJECT_MCP_HTTP_AUTH_PASSWORD")
    if not username or not password:
        return {}

    import base64

    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


async def _openproject_post_tool(path: str, params: dict[str, Any]) -> Any:
    base = _openproject_mcp_base_url()
    if not base:
        return {"_error": "missing_OPENPROJECT_MCP_URL"}
    url = f"{base}{path}"
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, params=params, headers=_openproject_mcp_auth_headers())
        except httpx.HTTPError as e:
            return {"_error": "request_failed", "detail": str(e) or repr(e)}
        if resp.status_code >= 400:
            return {
                "_error": "http_error",
                "status_code": resp.status_code,
                "response_text": resp.text,
            }
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.text}


def _parse_iso_datetime_any(value: Any):
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    from datetime import datetime

    try:
        normalized = s.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def _extract_title(value: Any) -> str | None:
    if isinstance(value, dict):
        t = value.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def _work_package_status(item: dict[str, Any]) -> str:
    links = item.get("_links") if isinstance(item.get("_links"), dict) else {}
    status = links.get("status")
    title = _extract_title(status)
    if title:
        return title
    v = item.get("status")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "Sin estado"


def _month_key(dt) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    import csv
    from io import StringIO

    if not rows:
        return b""
    fieldnames: list[str] = sorted({k for r in rows for k in r.keys()})
    s = StringIO()
    writer = csv.DictWriter(s, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    return s.getvalue().encode("utf-8")


async def _fetch_openproject_dashboard_data(project_id: int, *, max_work_packages: int = 300) -> dict[str, Any]:
    project = await _openproject_post_tool("/tools/get_project", {"project_id": project_id})
    memberships = await _openproject_post_tool(
        "/tools/list_memberships",
        {"project_id": project_id, "page_size": 200, "offset": 1},
    )
    work_packages = await _openproject_post_tool(
        "/tools/list_work_packages",
        {"project_id": project_id, "status": "all", "page_size": max_work_packages, "offset": 1},
    )
    return {"project": project, "memberships": memberships, "work_packages": work_packages}


def _render_openproject_dashboard(payload: dict[str, Any]) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        st.warning(
            "Para ver gráficos necesitas Plotly instalado. "
            "Instala `plotly` (y opcional `kaleido` para descargar PNG)."
        )
        st.json(payload)
        return

    project = payload.get("project") if isinstance(payload.get("project"), dict) else {}
    if project.get("_error"):
        st.error(f"Error obteniendo proyecto: {project}")
        return

    project_name = project.get("name") or project.get("identifier") or f"Proyecto {project.get('id')}"
    st.subheader(f"Dashboard: {project_name}")

    wps = payload.get("work_packages") if isinstance(payload.get("work_packages"), dict) else {}
    wp_embedded = wps.get("_embedded") if isinstance(wps.get("_embedded"), dict) else {}
    wp_elements = wp_embedded.get("elements") if isinstance(wp_embedded.get("elements"), list) else []
    wp_items = [e for e in wp_elements if isinstance(e, dict)]

    wp_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    created_by_month: dict[str, int] = {}

    for it in wp_items:
        status = _work_package_status(it)
        status_counts[status] = status_counts.get(status, 0) + 1

        created_at = it.get("createdAt")
        dt = _parse_iso_datetime_any(created_at)
        if dt:
            key = _month_key(dt)
            created_by_month[key] = created_by_month.get(key, 0) + 1

        wp_rows.append(
            {
                "id": it.get("id"),
                "subject": it.get("subject"),
                "status": status,
                "createdAt": created_at,
                "updatedAt": it.get("updatedAt"),
                "startDate": it.get("startDate") or it.get("derivedStartDate"),
                "percentageDone": it.get("percentageDone"),
            }
        )

    memberships = payload.get("memberships") if isinstance(payload.get("memberships"), dict) else {}
    m_embedded = memberships.get("_embedded") if isinstance(memberships.get("_embedded"), dict) else {}
    m_elements = m_embedded.get("elements") if isinstance(m_embedded.get("elements"), list) else []
    m_items = [e for e in m_elements if isinstance(e, dict)]

    role_counts: dict[str, int] = {}
    member_rows: list[dict[str, Any]] = []

    def extract_id_from_href(href: Any) -> int | None:
        if not isinstance(href, str):
            return None
        parts = [p for p in href.strip("/").split("/") if p]
        if not parts:
            return None
        try:
            return int(parts[-1])
        except Exception:
            return None

    for el in m_items:
        links = el.get("_links") if isinstance(el.get("_links"), dict) else {}
        principal = links.get("principal") if isinstance(links.get("principal"), dict) else {}
        principal_title = _extract_title(principal) or ""
        principal_id = extract_id_from_href(principal.get("href"))

        roles_link = links.get("roles")
        roles: list[str] = []
        if isinstance(roles_link, list):
            for r in roles_link:
                t = _extract_title(r)
                if t:
                    roles.append(t)
        elif isinstance(roles_link, dict):
            t = _extract_title(roles_link)
            if t:
                roles.append(t)

        if not roles:
            roles = ["Sin rol"]

        for r in roles:
            role_counts[r] = role_counts.get(r, 0) + 1

        member_rows.append(
            {
                "principal_id": principal_id,
                "principal_title": principal_title,
                "roles": ", ".join(roles),
                "membership_id": el.get("id"),
            }
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Work packages (muestra)", len(wp_rows))
    c2.metric("Miembros (muestra)", len(member_rows))
    c3.metric("Estados distintos", len(status_counts))

    if status_counts:
        df_status = [{"status": k, "count": v} for k, v in sorted(status_counts.items(), key=lambda x: -x[1])]
        fig_status = px.bar(df_status, x="status", y="count", title="Work packages por estado")
        st.plotly_chart(fig_status, use_container_width=True)

        st.download_button(
            "Descargar datos (work_packages.csv)",
            data=_csv_bytes(wp_rows),
            file_name=f"openproject_{project.get('id')}_work_packages.csv",
            mime="text/csv",
            use_container_width=True,
        )

        html = fig_status.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "Descargar gráfico (HTML)",
            data=html,
            file_name=f"openproject_{project.get('id')}_wp_por_estado.html",
            mime="text/html",
            use_container_width=True,
        )

        try:
            png = fig_status.to_image(format="png")  # requires kaleido
            st.download_button(
                "Descargar gráfico (PNG)",
                data=png,
                file_name=f"openproject_{project.get('id')}_wp_por_estado.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption("Para descargar PNG instala `kaleido`.")

    if created_by_month:
        df_month = [{"month": k, "count": v} for k, v in sorted(created_by_month.items())]
        fig_month = px.line(df_month, x="month", y="count", markers=True, title="Work packages creados por mes")
        st.plotly_chart(fig_month, use_container_width=True)

    if role_counts:
        df_roles = [{"role": k, "count": v} for k, v in sorted(role_counts.items(), key=lambda x: -x[1])]
        fig_roles = px.bar(df_roles, x="role", y="count", title="Distribución de roles (membresías)")
        st.plotly_chart(fig_roles, use_container_width=True)

        st.download_button(
            "Descargar datos (members.csv)",
            data=_csv_bytes(member_rows),
            file_name=f"openproject_{project.get('id')}_members.csv",
            mime="text/csv",
            use_container_width=True,
        )

    import json

    summary = {
        "project_id": project.get("id"),
        "work_packages_sample": len(wp_rows),
        "members_sample": len(member_rows),
        "status_counts": status_counts,
        "created_by_month": created_by_month,
        "role_counts": role_counts,
    }
    st.download_button(
        "Descargar resumen (JSON)",
        data=json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"openproject_{project.get('id')}_summary.json",
        mime="application/json",
        use_container_width=True,
    )


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # Initialize voice manager (once per session)
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # Clear saved audio when starting new chat
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)
            # Audio toggle with callback: clears cached audio when toggled off
            enable_audio = st.toggle(
                "Enable audio generation",
                value=True,
                disabled=not voice or not voice.tts,
                help="Configure VOICE_TTS_PROVIDER in .env to enable"
                if not voice or not voice.tts
                else None,
                on_change=lambda: st.session_state.pop("last_audio", None)
                if not st.session_state.get("enable_audio", True)
                else None,
                key="enable_audio",
            )

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        if agent_client.agent == OPENPROJECT_AGENT_ID:
            with st.expander("📊 OpenProject Dashboard", expanded=False):
                st.caption(
                    "Genera gráficos desde OpenProject vía `OPENPROJECT_MCP_URL`. "
                    "Puedes descargar datos (CSV/JSON) y el gráfico (HTML/PNG si tienes kaleido)."
                )
                base_url = _openproject_mcp_base_url()
                if not base_url:
                    st.error("Falta configurar `OPENPROJECT_MCP_URL` en tu `.env`.")
                else:
                    st.code(base_url, language="text")

                project_id = st.number_input("Project ID", min_value=1, step=1, value=1)
                max_wps = st.slider(
                    "Máx. work packages a cargar", min_value=50, max_value=500, value=300, step=50
                )
                if st.button("Generar gráficos", use_container_width=True):
                    with st.spinner("Cargando datos desde OpenProject MCP..."):
                        payload = await _fetch_openproject_dashboard_data(
                            int(project_id), max_work_packages=int(max_wps)
                        )
                    st.session_state["openproject_dashboard_payload"] = payload

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if agent_client.agent == OPENPROJECT_AGENT_ID and "openproject_dashboard_payload" in st.session_state:
        _render_openproject_dashboard(st.session_state["openproject_dashboard_payload"])

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Render saved audio for the last AI message (if it exists)
    # This ensures audio persists across st.rerun() calls
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # Generate new message if the user provided new input
    # Use voice manager if available, otherwise fall back to regular input
    # REQUIRED: Set VOICE_STT_PROVIDER, VOICE_TTS_PROVIDER, OPENAI_API_KEY
    # in app .env (NOT service .env) to enable voice features.
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # Generate TTS audio for streaming response
                # Note: draw_messages() stores the final message in st.session_state.messages
                # and the container reference in st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # Only generate audio for AI responses with content
                    if last_msg.type == "ai" and last_msg.content:
                        # Use audio_only=True since text was already streamed by draw_messages()
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # Render AI response with optional voice
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # Use different labels for transfer vs regular tool calls
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # Expect one ToolMessage for each tool call.
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # Only non-transfer tool calls reach this point
                            status = call_results[tool_call["id"]]
                            status.write("Input:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
