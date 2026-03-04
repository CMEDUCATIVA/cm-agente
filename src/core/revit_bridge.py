import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import WebSocket


class RevitBridgeError(RuntimeError):
    """Base exception for Revit bridge operations."""


class RevitBridgeSessionNotFound(RevitBridgeError):
    """Raised when a workstation session is not connected."""


@dataclass
class RevitBridgeSession:
    workstation_id: str
    websocket: WebSocket
    connected_at: datetime
    last_seen_at: datetime


class RevitBridgeManager:
    """Tracks Revit websocket sessions and dispatches command/response messages."""

    def __init__(self) -> None:
        self._sessions: dict[str, RevitBridgeSession] = {}
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    async def register(self, workstation_id: str, websocket: WebSocket) -> None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._sessions[workstation_id] = RevitBridgeSession(
                workstation_id=workstation_id,
                websocket=websocket,
                connected_at=now,
                last_seen_at=now,
            )

    async def unregister(self, workstation_id: str) -> None:
        async with self._lock:
            self._sessions.pop(workstation_id, None)

    async def touch(self, workstation_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(workstation_id)
            if session:
                session.last_seen_at = datetime.now(timezone.utc)

    async def list_sessions(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "workstation_id": s.workstation_id,
                    "connected_at": s.connected_at.isoformat(),
                    "last_seen_at": s.last_seen_at.isoformat(),
                }
                for s in self._sessions.values()
            ]

    async def send_command(
        self,
        workstation_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: int = 120,
    ) -> Any:
        async with self._lock:
            session = self._sessions.get(workstation_id)
            if not session:
                raise RevitBridgeSessionNotFound(
                    f"Revit session '{workstation_id}' is not connected."
                )
            websocket = session.websocket

        command_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[command_id] = future
        try:
            await websocket.send_json(
                {
                    "type": "command",
                    "command_id": command_id,
                    "method": method,
                    "params": params or {},
                }
            )
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result
        finally:
            self._pending.pop(command_id, None)

    async def handle_incoming_message(self, workstation_id: str, message: dict[str, Any]) -> None:
        await self.touch(workstation_id)
        msg_type = str(message.get("type") or "")
        if msg_type == "response":
            command_id = str(message.get("command_id") or "")
            if not command_id:
                return
            future = self._pending.get(command_id)
            if not future or future.done():
                return
            if bool(message.get("success", True)):
                future.set_result(message.get("result"))
            else:
                err = message.get("error") or "Unknown Revit plugin error"
                future.set_exception(RevitBridgeError(str(err)))


revit_bridge_manager = RevitBridgeManager()

