from typing import Any

from langchain_core.tools import BaseTool, tool

from core.revit_client import revit_plugin_client


async def _exec_revit_command(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(params or {})
    workstation_id = payload.pop("__workstation_id", None)
    result = await revit_plugin_client.send_command(method, payload, workstation_id=workstation_id)
    return {"method": method, "result": result}


@tool("Revit_SayHello")
async def revit_say_hello() -> dict[str, Any]:
    """Connection test. Calls say_hello inside Revit."""
    return await _exec_revit_command("say_hello")


@tool("Revit_GetAvailableFamilyTypes")
async def revit_get_available_family_types(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call get_available_family_types. Pass raw payload in params when needed."""
    return await _exec_revit_command("get_available_family_types", params)


@tool("Revit_GetCurrentViewElements")
async def revit_get_current_view_elements(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call get_current_view_elements. Pass raw payload in params when needed."""
    return await _exec_revit_command("get_current_view_elements", params)


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
async def revit_create_point_based_element(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_point_based_element with payload: {'data': [...]}."""
    return await _exec_revit_command("create_point_based_element", {"data": data})


@tool("Revit_CreateLineBasedElement")
async def revit_create_line_based_element(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_line_based_element with payload: {'data': [...]}."""
    return await _exec_revit_command("create_line_based_element", {"data": data})


@tool("Revit_CreateSurfaceBasedElement")
async def revit_create_surface_based_element(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_surface_based_element with payload: {'data': [...]}."""
    return await _exec_revit_command("create_surface_based_element", {"data": data})


@tool("Revit_ColorSplash")
async def revit_color_splash(params: dict[str, Any]) -> dict[str, Any]:
    """Call color_splash with raw params."""
    return await _exec_revit_command("color_splash", params)


@tool("Revit_TagWalls")
async def revit_tag_walls(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call tag_walls. Pass raw payload in params when needed."""
    return await _exec_revit_command("tag_walls", params)


@tool("Revit_DeleteElement")
async def revit_delete_element(element_ids: list[int | str]) -> dict[str, Any]:
    """Call delete_element with payload: {'elementIds': [...]}."""
    payload = {"elementIds": [str(i) for i in element_ids]}
    return await _exec_revit_command("delete_element", payload)


@tool("Revit_AiElementFilter")
async def revit_ai_element_filter(params: dict[str, Any]) -> dict[str, Any]:
    """Call ai_element_filter with raw params. Usually payload is {'data': {...}}."""
    return await _exec_revit_command("ai_element_filter", params)


@tool("Revit_OperateElement")
async def revit_operate_element(data: dict[str, Any]) -> dict[str, Any]:
    """Call operate_element with payload: {'data': {...}}."""
    return await _exec_revit_command("operate_element", {"data": data})


@tool("Revit_ExportRoomData")
async def revit_export_room_data(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call export_room_data. Pass raw payload in params when needed."""
    return await _exec_revit_command("export_room_data", params)


@tool("Revit_GetMaterialQuantities")
async def revit_get_material_quantities(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call get_material_quantities. Pass raw payload in params when needed."""
    return await _exec_revit_command("get_material_quantities", params)


@tool("Revit_AnalyzeModelStatistics")
async def revit_analyze_model_statistics(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call analyze_model_statistics. Pass raw payload in params when needed."""
    return await _exec_revit_command("analyze_model_statistics", params)


@tool("Revit_CreateGrid")
async def revit_create_grid(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_grid with payload: {'data': [...]}."""
    return await _exec_revit_command("create_grid", {"data": data})


@tool("Revit_CreateStructuralFramingSystem")
async def revit_create_structural_framing_system(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_structural_framing_system with payload: {'data': [...]}."""
    return await _exec_revit_command("create_structural_framing_system", {"data": data})


@tool("Revit_CreateRoom")
async def revit_create_room(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_room with payload: {'data': [...]}."""
    return await _exec_revit_command("create_room", {"data": data})


@tool("Revit_TagRooms")
async def revit_tag_rooms(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call tag_rooms. Pass raw payload in params when needed."""
    return await _exec_revit_command("tag_rooms", params)


@tool("Revit_CreateLevel")
async def revit_create_level(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_level with payload: {'data': [...]}."""
    return await _exec_revit_command("create_level", {"data": data})


@tool("Revit_SendCodeToRevit")
async def revit_send_code_to_revit(code: str, parameters: list[Any] | None = None) -> dict[str, Any]:
    """Call send_code_to_revit with dynamic C# code and optional parameters."""
    payload = {"code": code, "parameters": parameters or []}
    return await _exec_revit_command("send_code_to_revit", payload)


@tool("Revit_CreateDimensions")
async def revit_create_dimensions(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Call create_dimensions with payload: {'data': [...]}."""
    return await _exec_revit_command("create_dimensions", {"data": data})


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
    revit_operate_element,
    revit_export_room_data,
    revit_get_material_quantities,
    revit_analyze_model_statistics,
    revit_create_grid,
    revit_create_structural_framing_system,
    revit_create_room,
    revit_tag_rooms,
    revit_create_level,
    revit_send_code_to_revit,
    revit_create_dimensions,
]
