import base64
import asyncio
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from core import settings


NO_DATA_TEXT = "Aún no hay datos"
NO_DESCRIPTION_TEXT = "Sin descripción"


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


def _extract_description_raw(value: Any, *, max_len: int = 2000) -> str | None:
    if isinstance(value, str):
        raw = value
    elif isinstance(value, dict):
        raw = value.get("raw") if isinstance(value.get("raw"), str) else None
        if raw is None and isinstance(value.get("html"), str):
            raw = value.get("html")
    else:
        raw = None

    if raw is None:
        return None
    if len(raw) > max_len:
        return raw[:max_len] + "...(truncated)"
    return raw


def _display_or_no_data(value: Any) -> Any:
    if value is None:
        return NO_DATA_TEXT
    if isinstance(value, str) and not value.strip():
        return NO_DATA_TEXT
    if isinstance(value, (list, dict)) and not value:
        return NO_DATA_TEXT
    return value


async def _count_work_packages(*, project_id: int, status: str = "all") -> tuple[int | None, dict[str, Any] | None]:
    """
    Best-effort count of work packages for a project.

    We rely on the MCP adapter returning a collection payload with a `total` field.
    We also pass `page_size=1` to reduce payload if the adapter supports it.
    """
    result = await _post_tool(
        "/tools/list_work_packages",
        params={
            "project_id": project_id,
            "status": status,
            "page_size": 1,
        },
    )
    if isinstance(result, dict) and result.get("_error"):
        return None, result

    if isinstance(result, dict) and isinstance(result.get("total"), int):
        return int(result["total"]), None

    if isinstance(result, dict):
        embedded = result.get("_embedded") if isinstance(result.get("_embedded"), dict) else {}
        items = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []
        total = result.get("total")
        if isinstance(total, int):
            return total, None
        return len(items), None

    return None, {"_error": "openproject_mcp_unexpected_response", "result": _safe_str(result)}


def _project_preview(item: dict[str, Any]) -> dict[str, Any]:
    description_raw = _extract_description_raw(item.get("description"))
    if description_raw is None or (isinstance(description_raw, str) and not description_raw.strip()):
        description_value: Any = NO_DESCRIPTION_TEXT
    else:
        description_value = description_raw
    return {
        "id": item.get("id"),
        "identifier": item.get("identifier"),
        "name": item.get("name"),
        "active": item.get("active"),
        "public": item.get("public"),
        "description_raw": description_value,
        "createdAt": _display_or_no_data(item.get("createdAt")),
        "updatedAt": _display_or_no_data(item.get("updatedAt")),
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
async def openproject_list_projects(active_only: bool = True, max_items: int = 10) -> dict[str, Any]:
    """
    List projects (10 items by default) with a compact preview.

    Includes: description, createdAt/updatedAt, and a best-effort count of work packages.
    """
    result = await _post_tool("/tools/list_projects", params={"active_only": active_only})
    if isinstance(result, dict) and result.get("_error"):
        return result

    if not isinstance(result, dict):
        return {"result": _safe_str(result)}

    embedded = result.get("_embedded") if isinstance(result.get("_embedded"), dict) else {}
    items = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []
    total = result.get("total", len(items))

    max_items = max(1, min(int(max_items), 50))
    preview_src = [i for i in items[:max_items] if isinstance(i, dict)]
    preview = [_project_preview(i) for i in preview_src]

    sem = asyncio.Semaphore(4)

    async def add_counts(project: dict[str, Any]) -> None:
        project_id = project.get("id")
        if not isinstance(project_id, int):
            project["work_packages_total"] = NO_DATA_TEXT
            project["work_packages_error"] = {"_error": "missing_project_id"}
            return
        async with sem:
            count, error = await _count_work_packages(project_id=project_id, status="all")
        project["work_packages_total"] = count if isinstance(count, int) else NO_DATA_TEXT
        if error:
            project["work_packages_error"] = error

    await asyncio.gather(*(add_counts(p) for p in preview))

    counted = sum(1 for p in preview if isinstance(p.get("work_packages_total"), int))

    truncated = False
    if isinstance(total, int):
        truncated = total > len(preview)
    else:
        truncated = len(items) > max_items

    return {
        "_type": result.get("_type", "Collection"),
        "total": total,
        "count": len(items),
        "preview_count": len(preview),
        "truncated": truncated,
        "work_packages_counts": {"counted": counted, "attempted": len(preview)},
        "summary": f"Mostrando {len(preview)} de {total} proyectos.",
        "items": preview,
        "note": f"Mostrando solo los primeros {max_items} proyectos. Si deseas un proyecto en específico, dime el nombre (o parte del nombre) y lo busco.",
    }


@tool("OpenProject_SearchProjects")
async def openproject_search_projects(
    name_contains: str, active_only: bool = True, max_items: int = 10
) -> dict[str, Any]:
    """Search projects by name (server doesn't support filtering, so we filter client-side)."""
    result = await _post_tool("/tools/list_projects", params={"active_only": active_only})
    if isinstance(result, dict) and result.get("_error"):
        return result

    if not isinstance(result, dict):
        return {"result": _safe_str(result)}

    embedded = result.get("_embedded") if isinstance(result.get("_embedded"), dict) else {}
    items = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []
    needle = name_contains.strip().lower()
    matches = [
        i
        for i in items
        if isinstance(i, dict) and isinstance(i.get("name"), str) and needle in i["name"].lower()
    ]

    max_items = max(1, min(int(max_items), 50))
    preview_src = matches[:max_items]
    preview = [_project_preview(i) for i in preview_src if isinstance(i, dict)]

    sem = asyncio.Semaphore(4)

    async def add_counts(project: dict[str, Any]) -> None:
        project_id = project.get("id")
        if not isinstance(project_id, int):
            project["work_packages_total"] = NO_DATA_TEXT
            project["work_packages_error"] = {"_error": "missing_project_id"}
            return
        async with sem:
            count, error = await _count_work_packages(project_id=project_id, status="all")
        project["work_packages_total"] = count if isinstance(count, int) else NO_DATA_TEXT
        if error:
            project["work_packages_error"] = error

    await asyncio.gather(*(add_counts(p) for p in preview))

    counted = sum(1 for p in preview if isinstance(p.get("work_packages_total"), int))

    return {
        "_type": "Collection",
        "total": len(matches),
        "count": len(matches),
        "preview_count": len(preview),
        "truncated": len(matches) > max_items,
        "work_packages_counts": {"counted": counted, "attempted": len(preview)},
        "summary": f"Mostrando {len(preview)} de {len(matches)} proyectos que coinciden.",
        "query": name_contains,
        "items": preview,
        "note": "Si no ves el proyecto, intenta con otra parte del nombre o solicita el ID del proyecto.",
    }


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
    openproject_search_projects,
    openproject_get_project,
    openproject_list_work_packages,
    openproject_get_work_package,
    openproject_create_work_package,
    openproject_update_work_package,
    openproject_list_users,
    openproject_get_user,
    openproject_rest_list_projects,
]
