import base64
import asyncio
from datetime import datetime
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


_MONTHS_ES = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}


def _format_date_dd_mmm_yyyy(value: Any) -> str | None:
    """
    Format ISO-like date/datetime strings to:
    - `DD MMM YYYY` (e.g., `16 Dic 2025`) when no time is present
    - `DD MMM YYYY HH:mm` (e.g., `16 Dic 2025 06:20`) when time is present

    Returns None when the input can't be parsed as a date.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    has_time = ("T" in s) or (":" in s) or (len(s) > 10)

    # Handle common OpenProject formats:
    # - 2025-12-16T06:20:28.683Z
    # - 2025-12-16T06:20:28Z
    # - 2025-12-16
    try:
        normalized = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        month = _MONTHS_ES.get(dt.month, str(dt.month))
        if has_time:
            return f"{dt.day:02d} {month} {dt.year:04d} {dt.hour:02d}:{dt.minute:02d}"
        return f"{dt.day:02d} {month} {dt.year:04d}"
    except Exception:
        try:
            d = datetime.strptime(s[:10], "%Y-%m-%d")
            month = _MONTHS_ES.get(d.month, str(d.month))
            return f"{d.day:02d} {month} {d.year:04d}"
        except Exception:
            return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        normalized = s.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def _summarize_text(text: str, *, max_len: int = 180) -> str:
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "…"


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


def _work_package_preview(item: dict[str, Any]) -> dict[str, Any]:
    description_raw = _extract_description_raw(item.get("description"), max_len=1200)
    if description_raw is None or (isinstance(description_raw, str) and not description_raw.strip()):
        description_value: Any = NO_DESCRIPTION_TEXT
    else:
        description_value = _summarize_text(description_raw, max_len=240)
    links = item.get("_links") if isinstance(item.get("_links"), dict) else {}
    type_title = _extract_link_title(links.get("type"))
    status_title = _extract_link_title(links.get("status"))
    author_name, author_email = _extract_author_fields(item, links)
    assignee_name = _extract_link_title(links.get("assignee"))
    responsible_name = _extract_link_title(links.get("responsible"))
    return {
        "id": item.get("id"),
        "type": type_title,
        "status": status_title,
        "author_name": author_name,
        "author_email": author_email,
        "assignee_name": assignee_name,
        "responsible_name": responsible_name,
        "subject": item.get("subject"),
        "description_raw": description_value,
        "createdAt": _display_date_or_no_data(item.get("createdAt")),
        "updatedAt": _display_date_or_no_data(item.get("updatedAt")),
        "overallCosts": _extract_cost_total(item),
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


def _display_date_or_no_data(value: Any) -> str:
    formatted = _format_date_dd_mmm_yyyy(value)
    if formatted is not None:
        return formatted
    return str(_display_or_no_data(value))


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
        description_value = _summarize_text(description_raw)
    return {
        "id": item.get("id"),
        "identifier": item.get("identifier"),
        "name": item.get("name"),
        "active": item.get("active"),
        "public": item.get("public"),
        "description_raw": description_value,
        "createdAt": _display_date_or_no_data(item.get("createdAt")),
        "updatedAt": _display_date_or_no_data(item.get("updatedAt")),
    }


def _render_projects_markdown(payload: dict[str, Any]) -> str:
    total = payload.get("total", NO_DATA_TEXT)
    preview_count = payload.get("preview_count", 0)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    truncated = bool(payload.get("truncated"))

    lines: list[str] = []
    lines.append(f"Proyectos (mostrando {preview_count} de {total})")
    lines.append("")

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or NO_DATA_TEXT
        project_id = item.get("id") or NO_DATA_TEXT
        identifier = item.get("identifier") or NO_DATA_TEXT
        created_at = item.get("createdAt") or NO_DATA_TEXT
        updated_at = item.get("updatedAt") or NO_DATA_TEXT
        description = item.get("description_raw") or NO_DESCRIPTION_TEXT
        wp_total = item.get("work_packages_total") if item.get("work_packages_total") is not None else NO_DATA_TEXT

        lines.append(f"{idx}. {name}  ")
        lines.append(f"   - ID: {project_id} | Identificador: {identifier}")
        lines.append(f"   - Creado: {created_at} | Última actualización: {updated_at}")
        lines.append(f"   - Descripción: {description}")
        lines.append(f"   - Paquetes de trabajo (total): {wp_total}")
        lines.append("")

    lines.append(f"Total de proyectos: {total}")
    if truncated:
        lines.append("")
        lines.append("Para ver un proyecto específico, dime el nombre (o parte del nombre) y lo busco.")
    return "\n".join(lines).strip()


def _render_work_packages_markdown(payload: dict[str, Any]) -> str:
    work_packages = payload.get("work_packages") if isinstance(payload.get("work_packages"), dict) else {}
    memberships = payload.get("memberships") if isinstance(payload.get("memberships"), dict) else {}
    items = work_packages.get("items") if isinstance(work_packages.get("items"), list) else []
    member_items = memberships.get("items") if isinstance(memberships.get("items"), list) else []

    total = work_packages.get("total", NO_DATA_TEXT)
    preview_count = work_packages.get("preview_count", len(items))
    truncated = bool(work_packages.get("truncated"))
    next_offset = work_packages.get("next_offset")
    offset = work_packages.get("offset")
    page_size = work_packages.get("pageSize")
    status = payload.get("status", NO_DATA_TEXT)
    project_id = payload.get("project_id", NO_DATA_TEXT)

    lines: list[str] = []
    lines.append(f"Paquetes de trabajo (mostrando {preview_count} de {total})")
    lines.append(f"- Proyecto ID: {project_id} | Estado: {status}")
    lines.append("")

    for item in items:
        if not isinstance(item, dict):
            continue
        wp_id = _escape_md(item.get("id"))
        wp_type = _escape_md(item.get("type"))
        wp_status = _escape_md(item.get("status"))
        author_name = _escape_md(item.get("author_name"))
        author_email = _escape_md(item.get("author_email"))
        assignee_name = _escape_md(item.get("assignee_name"))
        responsible_name = _escape_md(item.get("responsible_name"))
        subject = _escape_md(item.get("subject"))
        description = _escape_md(item.get("description_raw") or NO_DESCRIPTION_TEXT)
        created = _escape_md(item.get("createdAt"))
        updated = _escape_md(item.get("updatedAt"))
        overall_costs = _escape_md(item.get("overallCosts"))

        if author_email and author_email != NO_DATA_TEXT:
            author_display = f"{author_name} <{author_email}>"
        else:
            author_display = author_name

        lines.append(f"Paquete de trabajo ID: {wp_id}")
        lines.append("")
        lines.append(f"| **Nombre** | {subject} |")
        lines.append("|---|---|")
        lines.append(f"| **ID** | {wp_id} |")
        lines.append(f"| **Tipo** | {wp_type} |")
        lines.append(f"| **Estado** | {wp_status} |")
        lines.append(f"| **Creador** | {author_display} |")
        lines.append(f"| **Asignado a** | {assignee_name} |")
        lines.append(f"| **Responsable** | {responsible_name} |")
        lines.append(f"| **Descripcion** | {description} |")
        lines.append(f"| **Creado** | {created} |")
        lines.append(f"| **Actualizado** | {updated} |")
        lines.append(f"| **Costo total** | {overall_costs} |")
        lines.append("")

    lines.append("")
    lines.append(f"Total de paquetes de trabajo: {total}")
    if truncated:
        extra = []
        if isinstance(offset, int) and isinstance(page_size, int):
            extra.append(f"Pagina actual: offset={offset}, pageSize={page_size}.")
        if isinstance(next_offset, int):
            extra.append(
                f"Para continuar: `OpenProject_ListWorkPackages(project_id={project_id}, status='{status}', max_items=20, work_packages_offset={next_offset})`."
            )
        if extra:
            lines.append(" ".join(extra))
        else:
            lines.append("Hay mas resultados. Si quieres continuar, dime 'continuar'.")

    # Summary table
    involved_names: list[str] = []
    for item in member_items:
        if not isinstance(item, dict):
            continue
        name = item.get("principal_title")
        if isinstance(name, str) and name.strip():
            involved_names.append(name.strip())
    if memberships and not involved_names and memberships.get("_error"):
        involved_names = []

    def _parse_cost_value(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            for ch in ("$", "€", "£", "USD", "EUR", "COP"):
                s = s.replace(ch, "")
            s = s.replace(",", "").strip()
            try:
                return float(s)
            except Exception:
                return None
        return None

    total_cost_value = 0.0
    cost_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        val = _parse_cost_value(item.get("overallCosts"))
        if val is not None:
            total_cost_value += val
            cost_count += 1

    total_cost_display: Any
    if cost_count > 0:
        total_cost_display = f"{total_cost_value:.2f}"
    else:
        total_cost_display = NO_DATA_TEXT

    involved_names_display = ", ".join(involved_names) if involved_names else NO_DATA_TEXT
    total_members_display = (
        memberships.get("total", len(member_items)) if memberships else NO_DATA_TEXT
    )

    lines.append("")
    lines.append("| Campo |  |  | Personas | Total |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Total de paquetes de trabajo |  | {total} |  | {total} |")
    lines.append(f"| Total de involucrados |  |  | {involved_names_display} | {total_members_display} |")
    lines.append(f"| Costo |  |  |  | {total_cost_display} |")

    return "\n".join(lines).strip()


def _escape_md(text: Any) -> str:
    if text is None:
        return NO_DATA_TEXT
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.replace("|", "\\|")


def _extract_id_from_href(href: Any) -> int | None:
    if not isinstance(href, str) or not href.strip():
        return None
    # Expected formats:
    # - /api/v3/projects/689
    # - /api/v3/users/450
    parts = [p for p in href.strip("/").split("/") if p]
    if not parts:
        return None
    tail = parts[-1]
    try:
        return int(tail)
    except Exception:
        return None


def _extract_status_title(value: Any) -> str:
    if isinstance(value, dict):
        title = value.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return NO_DATA_TEXT


def _extract_link_title(value: Any) -> str:
    if isinstance(value, dict):
        title = value.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return NO_DATA_TEXT


def _extract_author_fields(item: dict[str, Any], links: dict[str, Any]) -> tuple[str, str]:
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    name = author.get("name") if isinstance(author.get("name"), str) else None
    if not name:
        name = author.get("firstName") if isinstance(author.get("firstName"), str) else None
        last = author.get("lastName") if isinstance(author.get("lastName"), str) else None
        if name and last:
            name = f"{name} {last}"
        elif last:
            name = last

    email = author.get("email") if isinstance(author.get("email"), str) else None
    if not email:
        email = author.get("login") if isinstance(author.get("login"), str) else None

    if not name:
        name = _extract_link_title(links.get("author"))

    return name or NO_DATA_TEXT, email or NO_DATA_TEXT


def _extract_titles_from_links(value: Any) -> list[str]:
    """
    Extract a list of `title` values from an OpenProject link or list of links.

    Examples:
    - {"href": "...", "title": "Admin"} -> ["Admin"]
    - [{"href": "...", "title": "A"}, {"href": "...", "title": "B"}] -> ["A", "B"]
    """
    titles: list[str] = []
    if isinstance(value, dict):
        t = value.get("title")
        if isinstance(t, str) and t.strip():
            titles.append(t.strip())
        return titles
    if isinstance(value, list):
        for it in value:
            if isinstance(it, dict):
                t = it.get("title")
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())
        return titles
    return titles


def _project_membership_preview(el: dict[str, Any]) -> dict[str, Any]:
    links = el.get("_links") if isinstance(el.get("_links"), dict) else {}
    principal = links.get("principal") if isinstance(links.get("principal"), dict) else {}
    principal_href = principal.get("href")
    principal_id = _extract_id_from_href(principal_href)
    principal_title = principal.get("title") if isinstance(principal.get("title"), str) else None

    # Roles are typically a list under _links.roles
    role_titles = _extract_titles_from_links(links.get("roles"))

    return {
        "membership_id": el.get("id"),
        "principal_id": principal_id if principal_id is not None else NO_DATA_TEXT,
        "principal_title": principal_title or NO_DATA_TEXT,
        "role_titles": role_titles if role_titles else [NO_DATA_TEXT],
    }


async def _list_project_memberships(
    *,
    project_id: int,
    max_memberships: int = 50,
    memberships_offset: int = 1,
) -> dict[str, Any]:
    """
    List project memberships (members involved in the project).

    Uses the MCP adapter tool: POST /tools/list_memberships with project filter.
    """
    max_memberships = max(1, min(int(max_memberships), 100))
    memberships_offset = max(1, int(memberships_offset))

    memberships_result = await _post_tool(
        "/tools/list_memberships",
        params={
            "project_id": project_id,
            "page_size": max_memberships,
            "offset": memberships_offset,
        },
    )
    if (
        isinstance(memberships_result, dict)
        and memberships_result.get("_error") == "openproject_mcp_http_error"
        and memberships_result.get("status_code") in (400, 404, 422)
    ):
        # Some adapters may use a different filter name (e.g., project).
        memberships_result = await _post_tool(
            "/tools/list_memberships",
            params={
                "project": project_id,
                "page_size": max_memberships,
                "offset": memberships_offset,
            },
        )

    if isinstance(memberships_result, dict) and memberships_result.get("_error"):
        return {
            "_error": memberships_result,
            "total": NO_DATA_TEXT,
            "preview_count": 0,
            "truncated": False,
            "items": [],
        }

    if not isinstance(memberships_result, dict):
        return {
            "_error": {"_error": "openproject_mcp_unexpected_response", "result": _safe_str(memberships_result)},
            "total": NO_DATA_TEXT,
            "preview_count": 0,
            "truncated": False,
            "items": [],
        }

    embedded = (
        memberships_result.get("_embedded")
        if isinstance(memberships_result.get("_embedded"), dict)
        else {}
    )
    elements = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []

    total = memberships_result.get("total", len(elements))
    page_size = memberships_result.get("pageSize", max_memberships)
    offset = memberships_result.get("offset", memberships_offset)

    items = [_project_membership_preview(el) for el in elements if isinstance(el, dict)]

    next_offset: int | None = None
    if isinstance(total, int) and isinstance(offset, int):
        current_end = offset + len(elements)
        if current_end <= total:
            next_offset = current_end + 1
        if next_offset and next_offset > total:
            next_offset = None

    return {
        "_type": memberships_result.get("_type", "Collection"),
        "total": total,
        "count": len(elements),
        "pageSize": page_size,
        "offset": offset,
        "preview_count": len(items),
        "truncated": bool(total) and isinstance(total, int) and total > (offset - 1 + len(items)),
        "next_offset": next_offset,
        "items": items,
    }


def _work_package_status_from_item(item: dict[str, Any]) -> str:
    links = item.get("_links") if isinstance(item.get("_links"), dict) else {}
    status = links.get("status") if isinstance(links.get("status"), dict) else None
    status_title = _extract_status_title(status)
    if status_title != NO_DATA_TEXT:
        return status_title
    # Some payloads might have a top-level status field.
    status_field = item.get("status")
    if isinstance(status_field, str) and status_field.strip():
        return status_field.strip()
    return NO_DATA_TEXT


def _extract_cost_total(item: dict[str, Any]) -> str:
    # Best-effort: different OpenProject setups expose costs differently (or not at all).
    # Prefer "overall" costs when available (labor+materials).
    for key in (
        "overallCosts",
        "overall_costs",
        "overallCost",
        "overall_cost",
    ):
        v = item.get(key)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for key in ("totalCost", "total_cost", "costTotal", "cost_total", "costsTotal"):
        v = item.get(key)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Some APIs include a costs object.
    costs = item.get("costs")
    if isinstance(costs, dict):
        for key in ("total", "amount", "sum"):
            v = costs.get(key)
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # Some OpenProject instances expose budgets/costs via custom fields or nested objects.
    # We do a conservative deep-scan for cost-like keys, preferring "overall" when present.
    def score_key(key: str) -> int:
        k = key.lower().replace("_", "")
        if "overallcost" in k or ("overall" in k and "cost" in k):
            return 3
        if "totalcost" in k or ("total" in k and "cost" in k):
            return 2
        if "materialcost" in k or ("material" in k and "cost" in k):
            return 1
        if "laborcost" in k or ("labor" in k and "cost" in k):
            return 1
        return 0

    def scan(obj: Any, *, depth: int = 0) -> tuple[int, str] | None:
        if depth > 3:
            return None
        if isinstance(obj, dict):
            best: tuple[int, str] | None = None
            for k, v in obj.items():
                key = str(k).lower()
                if any(token in key for token in ("cost", "costo", "budget", "presupuesto", "price", "amount")):
                    sc = score_key(str(k))
                    if isinstance(v, (int, float)):
                        cand = (sc, str(v))
                        if best is None or cand[0] > best[0]:
                            best = cand
                    if isinstance(v, str) and v.strip():
                        cand = (sc, v.strip())
                        if best is None or cand[0] > best[0]:
                            best = cand
                    if isinstance(v, dict):
                        for kk in ("total", "amount", "sum", "value"):
                            vv = v.get(kk)
                            if isinstance(vv, (int, float)):
                                cand = (sc, str(vv))
                                if best is None or cand[0] > best[0]:
                                    best = cand
                            if isinstance(vv, str) and vv.strip():
                                cand = (sc, vv.strip())
                                if best is None or cand[0] > best[0]:
                                    best = cand

                nested = scan(v, depth=depth + 1)
                if nested is not None:
                    if best is None or nested[0] > best[0]:
                        best = nested
            return best
        elif isinstance(obj, list):
            best: tuple[int, str] | None = None
            for v in obj[:50]:
                nested = scan(v, depth=depth + 1)
                if nested is not None and (best is None or nested[0] > best[0]):
                    best = nested
            return best
        return None

    nested_cost = scan(item)
    if nested_cost is not None:
        return nested_cost[1]

    return NO_DATA_TEXT


def _extract_project_fields(result: dict[str, Any]) -> dict[str, Any]:
    # Full description for a single project can be important; keep it as complete as possible.
    # We still cap at a very high value to avoid extreme payloads blowing up responses.
    max_len = 200_000
    description_raw = _extract_description_raw(result.get("description"), max_len=max_len)
    if description_raw is None or not description_raw.strip():
        description_value: Any = NO_DESCRIPTION_TEXT
    else:
        description_value = description_raw

    description_truncated = isinstance(description_raw, str) and description_raw.endswith("...(truncated)")
    return {
        "id": result.get("id"),
        "identifier": result.get("identifier"),
        "name": result.get("name"),
        "active": result.get("active"),
        "public": result.get("public"),
        "createdAt": _display_date_or_no_data(result.get("createdAt")),
        "updatedAt": _display_date_or_no_data(result.get("updatedAt")),
        "description_raw": description_value,
        "description_truncated": description_truncated,
    }


def _render_project_detail_markdown(payload: dict[str, Any]) -> str:
    project = payload.get("project") if isinstance(payload.get("project"), dict) else {}
    work_packages = payload.get("work_packages") if isinstance(payload.get("work_packages"), dict) else {}
    wp_items = work_packages.get("items") if isinstance(work_packages.get("items"), list) else []
    memberships = payload.get("memberships") if isinstance(payload.get("memberships"), dict) else {}
    member_items = memberships.get("items") if isinstance(memberships.get("items"), list) else []

    name = project.get("name") or NO_DATA_TEXT
    project_id = project.get("id") or NO_DATA_TEXT
    created_at = project.get("createdAt") or NO_DATA_TEXT
    updated_at = project.get("updatedAt") or NO_DATA_TEXT
    description = project.get("description_raw") or NO_DESCRIPTION_TEXT
    description_truncated = bool(project.get("description_truncated"))

    wp_total = work_packages.get("total", NO_DATA_TEXT)
    wp_preview_count = work_packages.get("preview_count", len(wp_items))
    wp_truncated = bool(work_packages.get("truncated"))
    wp_next_offset = work_packages.get("next_offset")
    wp_offset = work_packages.get("offset")
    wp_page_size = work_packages.get("pageSize")

    lines: list[str] = []
    lines.append(f"Proyecto: {name}")
    lines.append(f"- ID: {project_id}")
    lines.append(f"- Creado: {created_at}")
    lines.append(f"- Última actualización: {updated_at}")
    lines.append(f"- Descripción: {description}")
    if description_truncated:
        lines.append(f"  (Nota: descripción truncada por tamaño; máx. {200_000} caracteres)")
    lines.append(f"- Paquetes de trabajo (total): {wp_total}")
    lines.append("")

    # Table
    lines.append("Paquetes de trabajo (vista previa):")
    lines.append("")
    lines.append("| ID | Estado | Nombre | Creado | Inicio | Última actualización | Costo total |")
    lines.append("|---:|---|---|---|---|---|---:|")

    for item in wp_items:
        if not isinstance(item, dict):
            continue
        wp_id = _escape_md(item.get("id"))
        status = _escape_md(item.get("status"))
        subject = _escape_md(item.get("subject"))
        created = _escape_md(item.get("createdAt"))
        start = _escape_md(item.get("startDate"))
        updated = _escape_md(item.get("updatedAt"))
        cost = _escape_md(item.get("cost_total"))
        lines.append(f"| {wp_id} | {status} | {subject} | {created} | {start} | {updated} | {cost} |")

    lines.append("")
    lines.append(f"Mostrando {wp_preview_count} de {wp_total} paquetes de trabajo.")
    if wp_truncated:
        extra = []
        if isinstance(wp_offset, int) and isinstance(wp_page_size, int):
            extra.append(f"Página actual: offset={wp_offset}, pageSize={wp_page_size}.")
        if isinstance(wp_next_offset, int):
            extra.append(
                f"Para continuar: solicita más con `OpenProject_GetProject(project_id={project_id}, work_packages_offset={wp_next_offset}, max_work_packages=20)`."
            )
        if extra:
            lines.append(" ".join(extra))
        else:
            lines.append(
                "Hay más resultados. Si quieres continuar, dime 'continuar' y te muestro los siguientes 20."
            )
        lines.append("También puedo filtrar por estado o por texto del nombre.")

    # Project memberships (people involved)
    if memberships:
        lines.append("")
        lines.append("Involucrados del proyecto:")

        members_total = memberships.get("total", NO_DATA_TEXT)
        members_preview_count = memberships.get("preview_count", len(member_items))
        members_truncated = bool(memberships.get("truncated"))
        members_next_offset = memberships.get("next_offset")
        members_offset = memberships.get("offset")
        members_page_size = memberships.get("pageSize")

        lines.append(f"Total de miembros: {members_total}")

        if memberships.get("_error"):
            lines.append(f"(Error obteniendo miembros: {memberships.get('_error')})")
        else:
            lines.append("")
            lines.append("| ID | Nombre | Rol | Membership ID |")
            lines.append("|---:|---|---|---:|")

            for item in member_items:
                if not isinstance(item, dict):
                    continue
                principal_id = _escape_md(item.get("principal_id"))
                principal_title = _escape_md(item.get("principal_title"))
                role_titles = item.get("role_titles")
                if isinstance(role_titles, list):
                    roles = ", ".join(
                        str(x) for x in role_titles if isinstance(x, str) and x.strip()
                    )
                elif isinstance(role_titles, str):
                    roles = role_titles.strip()
                else:
                    roles = ""
                roles = _escape_md(roles or NO_DATA_TEXT)
                membership_id = _escape_md(item.get("membership_id"))

                lines.append(f"| {principal_id} | {principal_title} | {roles} | {membership_id} |")

            lines.append("")
            lines.append(f"Mostrando {members_preview_count} de {members_total} miembros.")
            if members_truncated:
                extra = []
                if isinstance(members_offset, int) and isinstance(members_page_size, int):
                    extra.append(
                        f"Página actual: offset={members_offset}, pageSize={members_page_size}."
                    )
                if isinstance(members_next_offset, int):
                    extra.append(
                        f"Para continuar: solicita más con `OpenProject_GetProject(project_id={project_id}, memberships_offset={members_next_offset}, max_memberships=20)`."
                    )
                if extra:
                    lines.append(" ".join(extra))
                else:
                    lines.append(
                        "Hay más miembros. Si quieres continuar, dime 'continuar' y te muestro los siguientes 20."
                    )

    return "\n".join(lines).strip()


def _render_user_detail_markdown(payload: dict[str, Any]) -> str:
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    memberships = payload.get("memberships") if isinstance(payload.get("memberships"), dict) else {}
    items = memberships.get("items") if isinstance(memberships.get("items"), list) else []

    user_id = user.get("id", NO_DATA_TEXT)
    first = user.get("firstName", NO_DATA_TEXT)
    last = user.get("lastName", NO_DATA_TEXT)
    email = user.get("email", NO_DATA_TEXT)

    projects_total = payload.get("projects_total", NO_DATA_TEXT)
    m_total = memberships.get("total", NO_DATA_TEXT)
    m_preview = memberships.get("preview_count", len(items))
    truncated = bool(memberships.get("truncated"))
    next_offset = memberships.get("next_offset")
    offset = memberships.get("offset")
    page_size = memberships.get("pageSize")

    lines: list[str] = []
    lines.append("Usuario")
    lines.append(f"- ID: {user_id}")
    lines.append(f"- Nombres: {first}")
    lines.append(f"- Apellidos: {last}")
    lines.append(f"- Email: {email}")
    lines.append("")
    lines.append(f"Total de proyectos involucrados: {projects_total}")
    lines.append("")
    lines.append("Proyectos (por membresía):")
    lines.append("")
    lines.append("| Proyecto | Membresía | Creado | Actualizado |")
    lines.append("|---|---:|---|---|")
    for it in items:
        if not isinstance(it, dict):
            continue
        project_name = _escape_md(it.get("project_title"))
        project_id = it.get("project_id")
        if isinstance(project_id, int):
            project_name = f"{project_name} (ID: {project_id})"
        membership_id = _escape_md(it.get("membership_id"))
        created = _escape_md(it.get("createdAt"))
        updated = _escape_md(it.get("updatedAt"))
        lines.append(f"| {project_name} | {membership_id} | {created} | {updated} |")

    lines.append("")
    lines.append(f"Mostrando {m_preview} de {m_total} membresías.")
    if truncated:
        extra = []
        if isinstance(offset, int) and isinstance(page_size, int):
            extra.append(f"Página actual: offset={offset}, pageSize={page_size}.")
        if isinstance(next_offset, int):
            extra.append(
                f"Para continuar: `OpenProject_GetUser(user_id={user_id}, memberships_offset={next_offset}, max_memberships=20)`."
            )
        if extra:
            lines.append(" ".join(extra))
        else:
            lines.append("Hay más resultados. Si quieres continuar, dime 'continuar' y te muestro los siguientes 20.")

    return "\n".join(lines).strip()


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

    payload = {
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
    payload["rendered"] = _render_projects_markdown(payload)
    return payload


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
async def openproject_get_project(
    project_id: int,
    max_work_packages: int = 20,
    work_packages_offset: int = 1,
    max_memberships: int = 50,
    memberships_offset: int = 1,
) -> dict[str, Any]:
    """
    Get project details by ID, including a preview table of work packages.

    Output includes:
    - Project fields (created/updated, full description).
    - Work packages total count + a table preview with key columns.
    """
    project_result = await _post_tool("/tools/get_project", params={"project_id": project_id})
    if isinstance(project_result, dict) and project_result.get("_error"):
        return project_result
    if not isinstance(project_result, dict):
        return {"result": _safe_str(project_result)}

    project = _extract_project_fields(project_result)

    memberships_payload = await _list_project_memberships(
        project_id=project_id,
        max_memberships=max_memberships,
        memberships_offset=memberships_offset,
    )

    max_work_packages = max(1, min(int(max_work_packages), 50))
    work_packages_offset = max(1, int(work_packages_offset))
    wps_result = await _post_tool(
        "/tools/list_work_packages",
        params={
            "project_id": project_id,
            "status": "all",
            "page_size": max_work_packages,
            "offset": work_packages_offset,
        },
    )
    if isinstance(wps_result, dict) and wps_result.get("_error"):
        payload = {
            "project": project,
            "work_packages": {
                "_error": wps_result,
                "total": NO_DATA_TEXT,
                "preview_count": 0,
                "truncated": False,
                "items": [],
            },
            "memberships": memberships_payload,
        }
        payload["rendered"] = _render_project_detail_markdown(payload)
        return payload

    if not isinstance(wps_result, dict):
        payload = {
            "project": project,
            "work_packages": {"total": NO_DATA_TEXT, "preview_count": 0, "truncated": False, "items": []},
            "memberships": memberships_payload,
        }
        payload["rendered"] = _render_project_detail_markdown(payload)
        return payload

    embedded = wps_result.get("_embedded") if isinstance(wps_result.get("_embedded"), dict) else {}
    elements = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []

    # Sort by newest first (createdAt, fallback updatedAt).
    def sort_key(item: Any) -> datetime:
        if not isinstance(item, dict):
            return datetime.min
        dt = _parse_iso_datetime(item.get("createdAt")) or _parse_iso_datetime(item.get("updatedAt"))
        return dt or datetime.min

    elements_sorted = sorted([e for e in elements if isinstance(e, dict)], key=sort_key, reverse=True)

    total = wps_result.get("total", len(elements_sorted))
    page_size = wps_result.get("pageSize", max_work_packages)
    offset = wps_result.get("offset", work_packages_offset)

    # Enrich each work package using get_work_package for better fields.
    sem = asyncio.Semaphore(4)

    async def enrich(wp_item: dict[str, Any]) -> dict[str, Any]:
        wp_id = wp_item.get("id")
        cost_from_list = _extract_cost_total(wp_item)
        start_date_list = wp_item.get("startDate") or wp_item.get("derivedStartDate")
        if not isinstance(wp_id, int):
            return {
                "id": wp_item.get("id"),
                "status": _work_package_status_from_item(wp_item),
                "subject": wp_item.get("subject"),
                "createdAt": _display_date_or_no_data(wp_item.get("createdAt")),
                "startDate": _display_date_or_no_data(start_date_list),
                "updatedAt": _display_date_or_no_data(wp_item.get("updatedAt")),
                "cost_total": cost_from_list,
            }
        async with sem:
            detailed = await _post_tool("/tools/get_work_package", params={"work_package_id": wp_id})
        if isinstance(detailed, dict) and detailed.get("_error"):
            base = wp_item
        elif isinstance(detailed, dict):
            base = detailed
        else:
            base = wp_item

        return {
            "id": base.get("id", wp_id),
            "status": _work_package_status_from_item(base),
            "subject": base.get("subject") or wp_item.get("subject"),
            "createdAt": _display_date_or_no_data(base.get("createdAt")),
            "startDate": _display_date_or_no_data(base.get("startDate") or base.get("derivedStartDate") or start_date_list),
            "updatedAt": _display_date_or_no_data(base.get("updatedAt")),
            # Prefer costs coming from list_work_packages (as requested); fallback to detailed item.
            "cost_total": cost_from_list if cost_from_list != NO_DATA_TEXT else _extract_cost_total(base),
        }

    preview_src = elements_sorted[:max_work_packages]
    wp_items = await asyncio.gather(*(enrich(e) for e in preview_src))

    next_offset: int | None = None
    if isinstance(total, int) and isinstance(offset, int):
        current_end = offset + len(preview_src)
        if current_end <= total:
            next_offset = current_end + 1
        if next_offset and next_offset > total:
            next_offset = None

    work_packages = {
        "_type": wps_result.get("_type", "Collection"),
        "total": total,
        "count": len(elements_sorted),
        "pageSize": page_size,
        "offset": offset,
        "preview_count": len(wp_items),
        "truncated": bool(total) and isinstance(total, int) and total > (offset - 1 + len(wp_items)),
        "next_offset": next_offset,
        "items": wp_items,
    }

    payload = {"project": project, "work_packages": work_packages, "memberships": memberships_payload}
    payload["rendered"] = _render_project_detail_markdown(payload)
    return payload


@tool("OpenProject_ListWorkPackages")
async def openproject_list_work_packages(
    project_id: int,
    status: str = "open",
    max_items: int = 50,
    work_packages_offset: int = 1,
    max_memberships: int = 50,
    memberships_offset: int = 1,
) -> dict[str, Any]:
    """
    List work packages by project ID with a compact preview and project memberships.

    Returns:
    - work_packages: total, pagination, and items with id/subject/description/createdAt/updatedAt/overallCosts
    - memberships: project members preview (involucrados)
    - rendered: Markdown summary for chat
    """
    max_items = max(1, min(int(max_items), 50))
    work_packages_offset = max(1, int(work_packages_offset))
    result = await _post_tool(
        "/tools/list_work_packages",
        params={
            "project_id": project_id,
            "status": status,
            "page_size": max_items,
            "offset": work_packages_offset,
        },
    )
    if isinstance(result, dict) and result.get("_error"):
        return result
    if not isinstance(result, dict):
        return {"result": _safe_str(result)}

    embedded = result.get("_embedded") if isinstance(result.get("_embedded"), dict) else {}
    elements = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []
    items = [e for e in elements if isinstance(e, dict)]

    total = result.get("total", len(items))
    page_size = result.get("pageSize", max_items)
    offset = result.get("offset", work_packages_offset)

    preview = [_work_package_preview(it) for it in items[:max_items]]

    next_offset: int | None = None
    if isinstance(total, int) and isinstance(offset, int):
        current_end = offset + len(preview)
        if current_end <= total:
            next_offset = current_end + 1
        if next_offset and next_offset > total:
            next_offset = None

    work_packages_payload = {
        "_type": result.get("_type", "Collection"),
        "total": total,
        "count": len(items),
        "pageSize": page_size,
        "offset": offset,
        "preview_count": len(preview),
        "truncated": bool(total) and isinstance(total, int) and total > (offset - 1 + len(preview)),
        "next_offset": next_offset,
        "items": preview,
    }

    memberships_payload = await _list_project_memberships(
        project_id=project_id,
        max_memberships=max_memberships,
        memberships_offset=memberships_offset,
    )

    payload = {
        "project_id": project_id,
        "status": status,
        "work_packages": work_packages_payload,
        "memberships": memberships_payload,
    }
    payload["rendered"] = _render_work_packages_markdown(payload)
    return payload


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
async def openproject_get_user(
    user_id: int, max_memberships: int = 20, memberships_offset: int = 1
) -> dict[str, Any]:
    """
    Get a user by ID and list their project memberships.

    Returns:
    - `user`: id, firstName, lastName, email
    - `projects_total`: unique projects count in the returned memberships (best-effort)
    - `memberships`: membership list with project title and membership created/updated
    - `rendered`: Markdown ready for chat
    """
    user_result = await _post_tool("/tools/get_user", params={"user_id": user_id})
    if isinstance(user_result, dict) and user_result.get("_error"):
        return user_result
    if not isinstance(user_result, dict):
        return {"result": _safe_str(user_result)}

    user = {
        "id": user_result.get("id"),
        "firstName": _display_or_no_data(user_result.get("firstName")),
        "lastName": _display_or_no_data(user_result.get("lastName")),
        "email": _display_or_no_data(user_result.get("email")),
    }

    max_memberships = max(1, min(int(max_memberships), 50))
    memberships_offset = max(1, int(memberships_offset))

    memberships_result = await _post_tool(
        "/tools/list_memberships",
        params={
            "user_id": user_id,
            "page_size": max_memberships,
            "offset": memberships_offset,
        },
    )
    if (
        isinstance(memberships_result, dict)
        and memberships_result.get("_error") == "openproject_mcp_http_error"
        and memberships_result.get("status_code") in (400, 404, 422)
    ):
        # Some MCP adapters might use a different filter name (e.g., principal_id).
        memberships_result = await _post_tool(
            "/tools/list_memberships",
            params={
                "principal_id": user_id,
                "page_size": max_memberships,
                "offset": memberships_offset,
            },
        )
    if isinstance(memberships_result, dict) and memberships_result.get("_error"):
        payload = {
            "user": user,
            "projects_total": NO_DATA_TEXT,
            "memberships": {"_error": memberships_result, "total": NO_DATA_TEXT, "preview_count": 0, "items": []},
        }
        payload["rendered"] = _render_user_detail_markdown(payload)
        return payload

    if not isinstance(memberships_result, dict):
        payload = {
            "user": user,
            "projects_total": NO_DATA_TEXT,
            "memberships": {"total": NO_DATA_TEXT, "preview_count": 0, "items": []},
        }
        payload["rendered"] = _render_user_detail_markdown(payload)
        return payload

    embedded = (
        memberships_result.get("_embedded")
        if isinstance(memberships_result.get("_embedded"), dict)
        else {}
    )
    elements = embedded.get("elements") if isinstance(embedded.get("elements"), list) else []

    total = memberships_result.get("total", len(elements))
    page_size = memberships_result.get("pageSize", max_memberships)
    offset = memberships_result.get("offset", memberships_offset)

    membership_items: list[dict[str, Any]] = []
    project_ids: set[int] = set()

    for el in elements:
        if not isinstance(el, dict):
            continue
        links = el.get("_links") if isinstance(el.get("_links"), dict) else {}
        project_link = links.get("project") if isinstance(links.get("project"), dict) else {}
        project_href = project_link.get("href")
        project_id = _extract_id_from_href(project_href)
        project_title = project_link.get("title") if isinstance(project_link.get("title"), str) else None
        if project_id is not None:
            project_ids.add(project_id)

        membership_items.append(
            {
                "membership_id": el.get("id"),
                "createdAt": _display_date_or_no_data(el.get("createdAt")),
                "updatedAt": _display_date_or_no_data(el.get("updatedAt")),
                "_updatedAt_raw": el.get("updatedAt"),
                "project_id": project_id,
                "project_title": project_title or NO_DATA_TEXT,
            }
        )

    # Sort memberships by newest update first.
    membership_items.sort(
        key=lambda x: _parse_iso_datetime(x.get("_updatedAt_raw")) or datetime.min, reverse=True
    )

    next_offset: int | None = None
    if isinstance(total, int) and isinstance(offset, int):
        current_end = offset + len(elements)
        if current_end <= total:
            next_offset = current_end + 1
        if next_offset and next_offset > total:
            next_offset = None

    memberships_payload = {
        "_type": memberships_result.get("_type", "Collection"),
        "total": total,
        "count": len(elements),
        "pageSize": page_size,
        "offset": offset,
        "preview_count": len(membership_items),
        "truncated": bool(total) and isinstance(total, int) and total > (offset - 1 + len(membership_items)),
        "next_offset": next_offset,
        "items": [{k: v for k, v in it.items() if not str(k).startswith("_")} for it in membership_items[:max_memberships]],
    }

    payload = {
        "user": user,
        "projects_total": len(project_ids),
        "memberships": memberships_payload,
    }
    payload["rendered"] = _render_user_detail_markdown(payload)
    return payload


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
