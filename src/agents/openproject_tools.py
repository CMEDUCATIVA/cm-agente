import base64
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from core import settings


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
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, params=params or {}, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()


async def _get_rest(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_mcp_base_url()}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params or {}, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()


@tool("OpenProject_TestConnection")
async def openproject_test_connection() -> dict[str, Any]:
    """Test connection to the OpenProject MCP HTTP adapter."""
    result = await _post_tool("/tools/test_connection")
    if isinstance(result, dict):
        return result
    return {"result": result}


@tool("OpenProject_ListProjects")
async def openproject_list_projects(active_only: bool = True) -> dict[str, Any]:
    """List projects (returns the full collection)."""
    result = await _post_tool("/tools/list_projects", params={"active_only": active_only})
    return result


@tool("OpenProject_GetProject")
async def openproject_get_project(project_id: int) -> dict[str, Any]:
    """Get project details by ID."""
    return await _post_tool("/tools/get_project", params={"project_id": project_id})


@tool("OpenProject_ListWorkPackages")
async def openproject_list_work_packages(project_id: int, status: str = "open") -> dict[str, Any]:
    """List work packages by project ID (default status=open)."""
    return await _post_tool(
        "/tools/list_work_packages",
        params={
            "project_id": project_id,
            "status": status,
        },
    )


@tool("OpenProject_GetWorkPackage")
async def openproject_get_work_package(work_package_id: int) -> dict[str, Any]:
    """Get work package details by ID."""
    return await _post_tool("/tools/get_work_package", params={"work_package_id": work_package_id})


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
    return await _post_tool("/tools/create_work_package", params=params)


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
    return await _post_tool("/tools/update_work_package", params=params)


@tool("OpenProject_ListUsers")
async def openproject_list_users(active_only: bool = True) -> dict[str, Any]:
    """List users (returns the full collection)."""
    return await _post_tool("/tools/list_users", params={"active_only": active_only})


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

