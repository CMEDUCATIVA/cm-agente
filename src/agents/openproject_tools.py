import base64
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from core import settings


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _compact_collection(
    result: Any,
    *,
    max_items: int = 50,
    fields: list[str] | None = None,
    title_field: str | None = None,
) -> dict[str, Any]:
    """
    Reduce OpenProject collection payloads to a small preview to avoid blowing up LLM context.

    The OpenProject MCP adapter often returns full collections (potentially thousands of items).
    """
    if not isinstance(result, dict):
        return {"result": _safe_str(result)}

    embedded = result.get("_embedded") if isinstance(result.get("_embedded"), dict) else {}
    items = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []

    total = result.get("total", len(items))
    count = len(items)
    max_items = max(1, min(int(max_items), 200))

    def pick(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"value": _safe_str(item)}
        picked: dict[str, Any] = {}
        if fields:
            for k in fields:
                if k in item:
                    picked[k] = item.get(k)
        else:
            picked["id"] = item.get("id")
        if title_field and title_field in item:
            picked[title_field] = item.get(title_field)
        return picked

    preview = [pick(i) for i in items[:max_items]]

    return {
        "_type": result.get("_type", "Collection"),
        "total": total,
        "count": count,
        "preview_count": len(preview),
        "truncated": count > max_items,
        "items": preview,
    }


def _summarize_work_package(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"result": _safe_str(result)}
    description = result.get("description")
    description_raw = description.get("raw") if isinstance(description, dict) else None
    if isinstance(description_raw, str) and len(description_raw) > 5000:
        description_raw = description_raw[:5000] + "...(truncated)"
    return {
        "id": result.get("id"),
        "subject": result.get("subject"),
        "description_raw": description_raw,
        "percentageDone": result.get("percentageDone"),
        "_links": result.get("_links"),
    }


def _mcp_base_url() -> str:
    if not settings.OPENPROJECT_MCP_URL:
        raise ValueError("OPENPROJECT_MCP_URL is not set")
    return settings.OPENPROJECT_MCP_URL.rstrip("/")


def _auth_headers() -> dict[str, str]:
    if not settings.OPENPROJECT_MCP_HTTP_AUTH_ENABLED:
        return {}
    if not settings.OPENPROJECT_MCP_HTTP_AUTH_USERNAME or not settings.OPENPROJECT_MCP_HTTP_AUTH_PASSWORD:
        raise ValueError(
            "OPENPROJECT_MCP_HTTP_AUTH_ENABLED=true requires OPENPROJECT_MCP_HTTP_AUTH_USERNAME and OPENPROJECT_MCP_HTTP_AUTH_PASSWORD"
        )
    raw = f"{settings.OPENPROJECT_MCP_HTTP_AUTH_USERNAME}:{settings.OPENPROJECT_MCP_HTTP_AUTH_PASSWORD.get_secret_value()}"
    token = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


async def _post_tool(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_mcp_base_url()}{path}"
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, params=params or {}, headers=_auth_headers())
        except httpx.HTTPError as e:
            return {
                "_error": "openproject_mcp_request_failed",
                "method": "POST",
                "url": url,
                "exception": type(e).__name__,
                "detail": str(e) or repr(e),
            }

        if resp.status_code >= 400:
            return {
                "_error": "openproject_mcp_http_error",
                "method": "POST",
                "url": url,
                "status_code": resp.status_code,
                "response_text": resp.text,
            }
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.text}


async def _get_rest(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_mcp_base_url()}{path}"
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params or {}, headers=_auth_headers())
        except httpx.HTTPError as e:
            return {
                "_error": "openproject_mcp_request_failed",
                "method": "GET",
                "url": url,
                "exception": type(e).__name__,
                "detail": str(e) or repr(e),
            }

        if resp.status_code >= 400:
            return {
                "_error": "openproject_mcp_http_error",
                "method": "GET",
                "url": url,
                "status_code": resp.status_code,
                "response_text": resp.text,
            }
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.text}


@tool("OpenProject_TestConnection")
async def openproject_test_connection() -> dict[str, Any]:
    """Test connection to the OpenProject MCP HTTP adapter."""
    result = await _post_tool("/tools/test_connection")
    if isinstance(result, dict):
        return result
    return {"result": result}


@tool("OpenProject_ListProjects")
async def openproject_list_projects(active_only: bool = True, max_items: int = 50) -> dict[str, Any]:
    """List projects (returns a compact preview to avoid huge prompts)."""
    result = await _post_tool("/tools/list_projects", params={"active_only": active_only})
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _compact_collection(
        result,
        max_items=max_items,
        fields=["id", "name", "identifier", "active", "public"],
        title_field="name",
    )


@tool("OpenProject_GetProject")
async def openproject_get_project(project_id: int) -> dict[str, Any]:
    """Get project details by ID."""
    return await _post_tool("/tools/get_project", params={"project_id": project_id})


@tool("OpenProject_ListWorkPackages")
async def openproject_list_work_packages(
    project_id: int, status: str = "open", max_items: int = 50
) -> dict[str, Any]:
    """List work packages by project ID (compact preview to avoid huge prompts)."""
    result = await _post_tool(
        "/tools/list_work_packages",
        params={
            "project_id": project_id,
            "status": status,
        },
    )
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _compact_collection(
        result,
        max_items=max_items,
        fields=["id", "subject", "percentageDone"],
        title_field="subject",
    )


@tool("OpenProject_GetWorkPackage")
async def openproject_get_work_package(work_package_id: int) -> dict[str, Any]:
    """Get work package details by ID."""
    result = await _post_tool("/tools/get_work_package", params={"work_package_id": work_package_id})
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _summarize_work_package(result)


@tool("OpenProject_CreateWorkPackage")
async def openproject_create_work_package(
    project_id: int,
    subject: str,
    type_id: int,
    description: str | None = None,
    priority_id: int | None = None,
    assignee_id: int | None = None,
) -> dict[str, Any]:
    """Create a work package."""
    params: dict[str, Any] = {
        "project_id": project_id,
        "subject": subject,
        "type_id": type_id,
    }
    if description is not None:
        params["description"] = description
    if priority_id is not None:
        params["priority_id"] = priority_id
    if assignee_id is not None:
        params["assignee_id"] = assignee_id
    result = await _post_tool("/tools/create_work_package", params=params)
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _summarize_work_package(result)


@tool("OpenProject_UpdateWorkPackage")
async def openproject_update_work_package(
    work_package_id: int,
    subject: str | None = None,
    description: str | None = None,
    status_id: int | None = None,
    priority_id: int | None = None,
    assignee_id: int | None = None,
    percentage_done: int | None = None,
) -> dict[str, Any]:
    """Update a work package."""
    params: dict[str, Any] = {"work_package_id": work_package_id}
    if subject is not None:
        params["subject"] = subject
    if description is not None:
        params["description"] = description
    if status_id is not None:
        params["status_id"] = status_id
    if priority_id is not None:
        params["priority_id"] = priority_id
    if assignee_id is not None:
        params["assignee_id"] = assignee_id
    if percentage_done is not None:
        params["percentage_done"] = percentage_done
    result = await _post_tool("/tools/update_work_package", params=params)
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _summarize_work_package(result)


@tool("OpenProject_ListUsers")
async def openproject_list_users(active_only: bool = True, max_items: int = 50) -> dict[str, Any]:
    """List users (compact preview)."""
    result = await _post_tool("/tools/list_users", params={"active_only": active_only})
    if isinstance(result, dict) and result.get("_error"):
        return result
    return _compact_collection(
        result,
        max_items=max_items,
        fields=["id", "name", "login", "status"],
        title_field="name",
    )


@tool("OpenProject_GetUser")
async def openproject_get_user(user_id: int) -> dict[str, Any]:
    """Get user details by ID."""
    return await _post_tool("/tools/get_user", params={"user_id": user_id})


@tool("OpenProject_RestListProjects")
async def openproject_rest_list_projects(active: bool = True) -> dict[str, Any]:
    """(REST alias) List projects via GET /api/v1/projects."""
    return await _get_rest("/api/v1/projects", params={"active": active})


openproject_tools: list[BaseTool] = [
    openproject_test_connection,
    openproject_list_projects,
    openproject_get_project,
    openproject_list_work_packages,
    openproject_get_work_package,
    openproject_create_work_package,
    openproject_update_work_package,
    openproject_list_users,
    openproject_get_user,
    openproject_rest_list_projects,
]
