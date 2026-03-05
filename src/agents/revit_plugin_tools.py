from typing import Any

from langchain_core.tools import BaseTool, tool

from core import settings
from core.revit_bridge import RevitBridgeSessionNotFound, revit_bridge_manager


async def _exec_revit_command(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(params or {})
    workstation_id = payload.pop("__workstation_id", None) or settings.REVIT_BRIDGE_DEFAULT_WORKSTATION_ID

    # Map legacy RPC names to the websocket bridge method names used by the plugin.
    method = {
        "say_hello": "Revit_SayHello",
        "get_current_view_elements": "Revit_GetCurrentViewElements",
        "get_current_view_info": "Revit_GetCurrentViewInfo",
        "get_selected_elements": "Revit_GetSelectedElements",
        "delete_element": "Revit_DeleteElement",
        "create_level": "Revit_CreateLevel",
        "create_grid": "Revit_CreateGrid",
        "create_room": "Revit_CreateRoom",
        "tag_rooms": "Revit_TagRooms",
        "tag_walls": "Revit_TagWalls",
        "ai_element_filter": "Revit_AiElementFilter",
        "get_element_by_id": "Revit_GetElementById",
        "operate_element": "Revit_OperateElement",
        "create_dimensions": "Revit_CreateDimensions",
        "send_code_to_revit": "Revit_SendCodeToRevit",
        "create_point_based_element": "Revit_CreatePointBasedElement",
        "create_line_based_element": "Revit_CreateLineBasedElement",
        "create_surface_based_element": "Revit_CreateSurfaceBasedElement",
        "color_splash": "Revit_ColorSplash",
        "create_structural_framing_system": "Revit_CreateStructuralFramingSystem",
        "create_wall": "Revit_CreateWall",
        "create_floor": "Revit_CreateFloor",
        "create_roof": "Revit_CreateRoof",
        "create_stair": "Revit_CreateStair",
        "get_stair_runs_landings": "Revit_GetStairRunsLandings",
        "create_stair_dimensions": "Revit_CreateStairDimensions",
    }.get(method, method)

    if not workstation_id:
        sessions = await revit_bridge_manager.list_sessions()
        if len(sessions) == 1:
            workstation_id = str(sessions[0]["workstation_id"])
        elif not sessions:
            raise RevitBridgeSessionNotFound("No Revit bridge sessions are connected.")
        else:
            raise RevitBridgeSessionNotFound(
                "Multiple Revit sessions connected. Set REVIT_BRIDGE_DEFAULT_WORKSTATION_ID or pass __workstation_id."
            )

    result = await revit_bridge_manager.send_command(
        workstation_id=workstation_id,
        method=method,
        params=payload,
        timeout_seconds=int(settings.REVIT_BRIDGE_COMMAND_TIMEOUT_SECONDS),
    )
    return {"method": method, "result": result}


@tool("Revit_SayHello")
async def revit_say_hello() -> dict[str, Any]:
    """Connection test. Calls say_hello inside Revit."""
    return await _exec_revit_command("say_hello")


@tool("Revit_GetAvailableFamilyTypes")
async def revit_get_available_family_types(
    limit: int = 100,
    category: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List family types, with optional limit/category filters."""
    payload = dict(params or {})
    payload.setdefault("limit", max(1, min(int(limit), 2000)))
    if category:
        payload.setdefault("category", str(category))
    return await _exec_revit_command("get_available_family_types", payload)


@tool("Revit_GetCurrentViewElements")
async def revit_get_current_view_elements(
    limit: int = 200,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get elements in active view with explicit limit."""
    payload = dict(params or {})
    payload.setdefault("limit", max(1, min(int(limit), 5000)))
    return await _exec_revit_command("get_current_view_elements", payload)


@tool("Revit_GetCurrentViewInfo")
async def revit_get_current_view_info(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call get_current_view_info. Pass raw payload in params when needed."""
    return await _exec_revit_command("get_current_view_info", params)


@tool("Revit_GetSelectedElements")
async def revit_get_selected_elements(limit: int = 100) -> dict[str, Any]:
    """Call get_selected_elements with optional limit."""
    safe_limit = max(1, min(int(limit), 1000))
    return await _exec_revit_command("get_selected_elements", {"limit": safe_limit})


@tool("Revit_CreatePointBasedElement")
async def revit_create_point_based_element(
    data: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call create_point_based_element with payload: {'data': [...]}."""
    payload = dict(params or {})
    if data:
        payload["data"] = data
    if "data" not in payload:
        raise ValueError("data is required (example: [{'x':1,'y':1,'z':0,'units':'m'}])")
    if not isinstance(payload["data"], list) or len(payload["data"]) == 0:
        raise ValueError("data must be a non-empty list")
    return await _exec_revit_command("create_point_based_element", payload)


@tool("Revit_CreateLineBasedElement")
async def revit_create_line_based_element(
    data: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call create_line_based_element with payload: {'data': [...]}."""
    payload = dict(params or {})
    if data:
        payload["data"] = data
    if "data" not in payload:
        raise ValueError("data is required (example: [{'start': {...}, 'end': {...}}])")
    if not isinstance(payload["data"], list) or len(payload["data"]) == 0:
        raise ValueError("data must be a non-empty list")
    return await _exec_revit_command("create_line_based_element", payload)


@tool("Revit_CreateSurfaceBasedElement")
async def revit_create_surface_based_element(
    data: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call create_surface_based_element with payload: {'data': [...]}."""
    payload = dict(params or {})
    if data:
        payload["data"] = data
    if "data" not in payload:
        raise ValueError("data is required (example: [{'host_element_id': 123456}])")
    if not isinstance(payload["data"], list) or len(payload["data"]) == 0:
        raise ValueError("data must be a non-empty list")
    return await _exec_revit_command("create_surface_based_element", payload)


@tool("Revit_ColorSplash")
async def revit_color_splash(
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call color_splash with raw params."""
    payload = dict(params or {})
    if data is not None and "data" not in payload:
        payload["data"] = data
    if not payload:
        raise ValueError(
            "params is required (example: {'data': {'category': 'Walls', 'color': '#FF6600', 'transparency': 20}})"
        )
    return await _exec_revit_command("color_splash", payload)


@tool("Revit_TagWalls")
async def revit_tag_walls(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call tag_walls. Pass raw payload in params when needed."""
    return await _exec_revit_command("tag_walls", params)


@tool("Revit_DeleteElement")
async def revit_delete_element(element_ids: list[int | str]) -> dict[str, Any]:
    """Delete one element by id (uses first id when multiple are provided)."""
    if not element_ids:
        raise ValueError("element_ids is required")
    payload = {"element_id": int(str(element_ids[0]))}
    return await _exec_revit_command("Revit_DeleteElement", payload)


@tool("Revit_AiElementFilter")
async def revit_ai_element_filter(
    category: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call ai_element_filter with raw params. Supports category shortcut."""
    payload = dict(params or {})
    if category and "data" not in payload:
        payload["data"] = {"category": category}
    if not payload:
        raise ValueError("params is required (example: {'data': {'category': 'Walls'}})")
    return await _exec_revit_command("ai_element_filter", payload)


@tool("Revit_GetElementById")
async def revit_get_element_by_id(
    element_id: int | str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get element info by id with a compact payload."""
    payload = dict(params or {})
    if element_id is not None:
        payload.setdefault("element_id", int(str(element_id)))
    if "element_id" not in payload:
        raise ValueError("element_id is required")
    return await _exec_revit_command("get_element_by_id", payload)


@tool("Revit_OperateElement")
async def revit_operate_element(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call operate_element with payload: {'data': {...}}."""
    if not data:
        raise ValueError("data is required (example: {'operation': 'delete', 'element_ids': [123]})")
    return await _exec_revit_command("operate_element", {"data": data})


@tool("Revit_ExportRoomData")
async def revit_export_room_data(
    limit: int = 200,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export room data with optional limit."""
    payload = dict(params or {})
    payload.setdefault("limit", max(1, min(int(limit), 5000)))
    return await _exec_revit_command("export_room_data", payload)


@tool("Revit_GetMaterialQuantities")
async def revit_get_material_quantities(
    max_elements: int = 1000,
    top_materials: int = 100,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get material quantities with explicit limits."""
    payload = dict(params or {})
    payload.setdefault("max_elements", max(1, min(int(max_elements), 20000)))
    payload.setdefault("top_materials", max(1, min(int(top_materials), 5000)))
    return await _exec_revit_command("get_material_quantities", payload)


@tool("Revit_AnalyzeModelStatistics")
async def revit_analyze_model_statistics(
    top_categories: int = 25,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze model statistics with explicit top_categories arg."""
    payload = dict(params or {})
    payload.setdefault("top_categories", max(1, min(int(top_categories), 200)))
    return await _exec_revit_command("analyze_model_statistics", payload)


@tool("Revit_CreateGrid")
async def revit_create_grid(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_grid with payload: {'data': [...]}."""
    return await _exec_revit_command("create_grid", {"data": data})


@tool("Revit_CreateStructuralFramingSystem")
async def revit_create_structural_framing_system(
    data: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call create_structural_framing_system with payload: {'data': [...]}."""
    payload = dict(params or {})
    if data:
        payload["data"] = data
    if "data" not in payload:
        raise ValueError("data is required (example: [{'start': {...}, 'end': {...}, 'level': 'Level 1'}])")
    if not isinstance(payload["data"], list) or len(payload["data"]) == 0:
        raise ValueError("data must be a non-empty list")
    return await _exec_revit_command("create_structural_framing_system", payload)


@tool("Revit_CreateWall")
async def revit_create_wall(
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a wall from start/end points with optional level/type/height."""
    payload = dict(params or {})
    if data:
        payload.setdefault("data", data)
    if "data" not in payload:
        raise ValueError(
            "data is required (example: {'start':{'x':0,'y':0,'z':0,'units':'m'},'end':{'x':6,'y':0,'z':0,'units':'m'},'level':'Level 1','height':3,'units':'m'})"
        )
    if not isinstance(payload["data"], dict):
        raise ValueError("data must be an object")
    return await _exec_revit_command("create_wall", payload)


@tool("Revit_CreateFloor")
async def revit_create_floor(
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a floor from a closed polygon in points[]."""
    payload = dict(params or {})
    if data:
        payload.setdefault("data", data)
    if "data" not in payload:
        raise ValueError(
            "data is required (example: {'points':[{'x':0,'y':0,'z':0,'units':'m'},{'x':6,'y':0,'z':0,'units':'m'},{'x':6,'y':4,'z':0,'units':'m'},{'x':0,'y':4,'z':0,'units':'m'}],'level':'Level 1'})"
        )
    if not isinstance(payload["data"], dict):
        raise ValueError("data must be an object")
    return await _exec_revit_command("create_floor", payload)


@tool("Revit_CreateRoof")
async def revit_create_roof(
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a footprint roof from a closed polygon in points[]."""
    payload = dict(params or {})
    if data:
        payload.setdefault("data", data)
    if "data" not in payload:
        raise ValueError(
            "data is required (example: {'points':[{'x':0,'y':0,'z':3,'units':'m'},{'x':6,'y':0,'z':3,'units':'m'},{'x':6,'y':4,'z':3,'units':'m'},{'x':0,'y':4,'z':3,'units':'m'}],'level':'Level 2'})"
        )
    if not isinstance(payload["data"], dict):
        raise ValueError("data must be an object")
    return await _exec_revit_command("create_roof", payload)


@tool("Revit_CreateStair")
async def revit_create_stair(
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a straight stair run between base/top levels."""
    payload = dict(params or {})
    if data:
        payload.setdefault("data", data)
    if "data" not in payload:
        raise ValueError(
            "data is required (example: {'base_level':'Level 1','top_level':'Level 2','start':{'x':0,'y':0,'z':0,'units':'m'},'end':{'x':3,'y':0,'z':0,'units':'m'},'width':1.2})"
        )
    if not isinstance(payload["data"], dict):
        raise ValueError("data must be an object")
    return await _exec_revit_command("create_stair", payload)


@tool("Revit_GetStairRunsLandings")
async def revit_get_stair_runs_landings(
    stair_id: int | str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get run/landing ids for a stair."""
    payload = dict(params or {})
    if stair_id is not None:
        payload.setdefault("stair_id", int(str(stair_id)))
    if "stair_id" not in payload:
        raise ValueError("stair_id is required")
    return await _exec_revit_command("get_stair_runs_landings", payload)


@tool("Revit_CreateStairDimensions")
async def revit_create_stair_dimensions(
    stair_id: int | str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a dimension between first two runs of a stair when possible."""
    payload = dict(params or {})
    if stair_id is not None:
        payload.setdefault("stair_id", int(str(stair_id)))
    if "stair_id" not in payload:
        raise ValueError("stair_id is required")
    return await _exec_revit_command("create_stair_dimensions", payload)


@tool("Revit_CreateRoom")
async def revit_create_room(data: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Call create_room with payload: {'data': [...]}."""
    if not data:
        raise ValueError("data is required and must include at least one room payload")
    return await _exec_revit_command("create_room", {"data": data})


@tool("Revit_TagRooms")
async def revit_tag_rooms(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call tag_rooms. Pass raw payload in params when needed."""
    return await _exec_revit_command("tag_rooms", params)


@tool("Revit_CreateLevel")
async def revit_create_level(
    name: str | None = None,
    elevation: float | None = None,
    units: str = "m",
    data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Create a level.
    Preferred args: name, elevation, units.
    Backward compatible with data=[{"name": "...", "elevation": 12.3, "units": "m"}].
    """
    if data:
        item = data[0] if data else {}
        name = name or item.get("name") or item.get("level_name")
        elevation = (
            elevation
            if elevation is not None
            else item.get("elevation", item.get("elevation_m", item.get("elevationM")))
        )
        units = item.get("units", units)

    if elevation is None:
        elevation = 0.0

    payload = {
        "name": name or "New Level",
        "elevation": float(elevation),
        "units": str(units or "m"),
    }
    return await _exec_revit_command("Revit_CreateLevel", payload)


@tool("Revit_SendCodeToRevit")
async def revit_send_code_to_revit(code: str, parameters: list[Any] | None = None) -> dict[str, Any]:
    """Call send_code_to_revit with dynamic C# code and optional parameters."""
    payload = {"code": code, "parameters": parameters or []}
    return await _exec_revit_command("send_code_to_revit", payload)


@tool("Revit_CreateDimensions")
async def revit_create_dimensions(data: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Call create_dimensions. If data is omitted, plugin uses current selection/context."""
    payload: dict[str, Any] = {}
    if data:
        payload["data"] = data
    return await _exec_revit_command("create_dimensions", payload)


revit_plugin_tools: list[BaseTool] = [
    revit_say_hello,
    revit_get_available_family_types,
    revit_get_current_view_elements,
    revit_get_current_view_info,
    revit_get_selected_elements,
    revit_create_point_based_element,
    revit_create_line_based_element,
    revit_create_surface_based_element,
    revit_color_splash,
    revit_tag_walls,
    revit_delete_element,
    revit_ai_element_filter,
    revit_get_element_by_id,
    revit_operate_element,
    revit_export_room_data,
    revit_get_material_quantities,
    revit_analyze_model_statistics,
    revit_create_grid,
    revit_create_structural_framing_system,
    revit_create_wall,
    revit_create_floor,
    revit_create_roof,
    revit_create_stair,
    revit_get_stair_runs_landings,
    revit_create_stair_dimensions,
    revit_create_room,
    revit_tag_rooms,
    revit_create_level,
    revit_send_code_to_revit,
    revit_create_dimensions,
]
