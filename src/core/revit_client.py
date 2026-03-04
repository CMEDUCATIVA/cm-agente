import asyncio
import json
import time
import uuid
from typing import Any

from core.revit_bridge import (
    RevitBridgeSessionNotFound,
    revit_bridge_manager,
)
from core.settings import settings


class RevitPluginError(RuntimeError):
    """Raised when the Revit plugin returns a JSON-RPC error."""


class RevitPluginClient:
    """Minimal JSON-RPC client for direct communication with the Revit plugin socket service."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def send_command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        workstation_id: str | None = None,
    ) -> Any:
        if settings.REVIT_BRIDGE_ENABLED:
            target_workstation = workstation_id or settings.REVIT_BRIDGE_DEFAULT_WORKSTATION_ID
            if target_workstation:
                try:
                    return await revit_bridge_manager.send_command(
                        workstation_id=target_workstation,
                        method=method,
                        params=params or {},
                        timeout_seconds=int(settings.REVIT_BRIDGE_COMMAND_TIMEOUT_SECONDS),
                    )
                except RevitBridgeSessionNotFound:
                    # Fallback to direct socket mode if requested workstation is not connected.
                    pass

        request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id,
        }
        timeout_s = float(settings.REVIT_PLUGIN_TIMEOUT_SECONDS)

        async with self._lock:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(settings.REVIT_PLUGIN_HOST, settings.REVIT_PLUGIN_PORT),
                timeout=timeout_s,
            )
            try:
                writer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                await asyncio.wait_for(writer.drain(), timeout=timeout_s)
                response = await self._read_response(reader, timeout_s)
            finally:
                writer.close()
                await writer.wait_closed()

        if not isinstance(response, dict):
            raise RevitPluginError("Invalid response from Revit plugin.")
        if response.get("id") != request_id:
            raise RevitPluginError("Mismatched response id from Revit plugin.")
        if response.get("error") is not None:
            err = response.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or "Unknown plugin error")
                code = err.get("code")
                raise RevitPluginError(f"Plugin error {code}: {message}")
            raise RevitPluginError(f"Plugin error: {err}")
        return response.get("result")

    async def _read_response(
        self, reader: asyncio.StreamReader, timeout_s: float
    ) -> dict[str, Any] | list[Any]:
        buffer = b""
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            remaining = max(0.1, timeout_s - elapsed)
            chunk = await asyncio.wait_for(reader.read(8192), timeout=remaining)
            if not chunk:
                break
            buffer += chunk
            try:
                decoded = buffer.decode("utf-8")
                return json.loads(decoded)
            except json.JSONDecodeError:
                continue
        raise TimeoutError("Timed out waiting for Revit plugin response.")


revit_plugin_client = RevitPluginClient()
