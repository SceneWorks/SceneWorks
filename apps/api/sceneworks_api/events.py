from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException


class EventHub:
    def __init__(self) -> None:
        self._subscribers: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers[queue] = asyncio.get_running_loop()
        await queue.put({"event": "ready", "data": {"status": "connected"}})
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.pop(queue, None)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        message = {"event": event, "data": data}
        for queue, loop in list(self._subscribers.items()):
            try:
                loop.call_soon_threadsafe(self._put_nowait, queue, message)
            except RuntimeError:
                self.unsubscribe(queue)

    def _put_nowait(self, queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            self.unsubscribe(queue)


def encode_sse(message: dict[str, Any]) -> str:
    event = message.get("event", "message")
    data = json.dumps(message.get("data", {}), separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


class EventTicketStore:
    def __init__(self, ttl_seconds: int = 30) -> None:
        self.ttl_seconds = ttl_seconds
        self._tickets: dict[str, float] = {}

    def issue(self) -> dict[str, Any]:
        self._prune()
        ticket = uuid4().hex
        expires_at = time.time() + self.ttl_seconds
        self._tickets[ticket] = expires_at
        return {"ticket": ticket, "expiresInSeconds": self.ttl_seconds}

    def consume(self, ticket: str) -> None:
        self._prune()
        expires_at = self._tickets.pop(ticket, None)
        if expires_at is None or expires_at < time.time():
            raise HTTPException(status_code=401, detail="Invalid or expired event stream ticket")

    def _prune(self) -> None:
        now = time.time()
        expired = [ticket for ticket, expires_at in self._tickets.items() if expires_at < now]
        for ticket in expired:
            self._tickets.pop(ticket, None)
